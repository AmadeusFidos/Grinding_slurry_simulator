using System.Diagnostics;
using System.Text;
using System.Text.Json;

namespace LapSim;

/// <summary>
/// Talks to the Python simulator over a pipe.
/// </summary>
/// <remarks>
/// The engine is kept alive as a child process rather than started per request.
/// Nearly all the cost of a request is importing numpy and scipy, so a
/// persistent process answers in milliseconds where a fresh one takes most of a
/// second -- the difference between a slider that tracks your finger and one
/// that stutters.
///
/// The protocol is one JSON object per line in each direction; see
/// <c>lapsim/api.py</c>.
/// </remarks>
public sealed class PythonBridge : IDisposable
{
    private readonly Process _process;
    private readonly SemaphoreSlim _turn = new(1, 1);
    private int _nextId;

    /// <summary>How this bridge was started, for the status bar.</summary>
    public string Description { get; }

    /// <summary>Whether the interpreter has matplotlib, so figures can be drawn.</summary>
    public bool CanPlot { get; private set; }

    public JsonElement Capabilities { get; private set; }

    private PythonBridge(Process process, string description)
    {
        _process = process;
        Description = description;
    }

    /// <summary>
    /// Find a working interpreter and start it, or throw with everything tried.
    /// </summary>
    /// <param name="repositoryRoot">
    /// Directory holding the <c>lapsim</c> package. Passed through PYTHONPATH
    /// rather than as a working directory, because Windows cannot use a UNC path
    /// such as <c>\\wsl.localhost\...</c> as a process working directory -- and
    /// that is exactly where this project lives when it is edited under WSL.
    /// </param>
    public static PythonBridge Start(string repositoryRoot)
    {
        var attempts = new List<string>();
        foreach (var candidate in Candidates())
        {
            try
            {
                var bridge = TryStart(candidate, repositoryRoot);
                if (bridge is not null)
                {
                    return bridge;
                }
                attempts.Add($"{candidate.Describe()}: no usable response");
            }
            catch (Exception error)
            {
                attempts.Add($"{candidate.Describe()}: {error.Message}");
            }
        }

        throw new InvalidOperationException(
            "Could not start the Python engine. Tried:" + Environment.NewLine
            + string.Join(Environment.NewLine, attempts.Select(a => "  " + a))
            + Environment.NewLine + Environment.NewLine
            + "Set the LAPSIM_PYTHON environment variable to the interpreter to use. "
            + "It needs numpy and scipy; matplotlib is optional and only enables "
            + "the rendered figures.");
    }

    private readonly record struct Candidate(string Executable, string[] Arguments)
    {
        public string Describe() => (Executable + " " + string.Join(' ', Arguments)).Trim();
    }

    private static IEnumerable<Candidate> Candidates()
    {
        var configured = Environment.GetEnvironmentVariable("LAPSIM_PYTHON");
        if (!string.IsNullOrWhiteSpace(configured))
        {
            yield return new Candidate(configured, []);
        }

        yield return new Candidate("py", ["-3"]);
        yield return new Candidate("python", []);
        yield return new Candidate("python3", []);

        // Last resort: the project may only be runnable inside WSL, where the
        // engine and its dependencies already live.
        yield return new Candidate("wsl.exe", ["--", "python3"]);
    }

    private static PythonBridge? TryStart(Candidate candidate, string repositoryRoot)
    {
        var info = new ProcessStartInfo
        {
            FileName = candidate.Executable,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardInputEncoding = Encoding.UTF8,
        };

        foreach (var argument in candidate.Arguments)
        {
            info.ArgumentList.Add(argument);
        }

        var viaWsl = candidate.Executable.Equals("wsl.exe", StringComparison.OrdinalIgnoreCase);
        info.ArgumentList.Add("-m");
        info.ArgumentList.Add("lapsim.api");

        var pythonPath = viaWsl ? ToWslPath(repositoryRoot) : repositoryRoot;
        info.Environment["PYTHONPATH"] = pythonPath;
        info.Environment["PYTHONUNBUFFERED"] = "1";
        if (viaWsl)
        {
            // WSL only forwards variables it is told to.
            info.Environment["WSLENV"] = "PYTHONPATH/p:PYTHONUNBUFFERED";
        }

        var process = Process.Start(info);
        if (process is null)
        {
            return null;
        }

        var bridge = new PythonBridge(process, candidate.Describe());
        try
        {
            var capabilities = bridge.RequestAsync("capabilities", new { })
                                     .GetAwaiter().GetResult();
            bridge.Capabilities = capabilities;
            bridge.CanPlot = capabilities.TryGetProperty("can_plot", out var flag)
                             && flag.GetBoolean();
            return bridge;
        }
        catch
        {
            bridge.Dispose();
            throw;
        }
    }

    /// <summary>Translate <c>\\wsl.localhost\Distro\home\me\x</c> to <c>/home/me/x</c>.</summary>
    private static string ToWslPath(string windowsPath)
    {
        if (windowsPath.StartsWith(@"\\wsl.localhost\", StringComparison.OrdinalIgnoreCase)
            || windowsPath.StartsWith(@"\\wsl$\", StringComparison.OrdinalIgnoreCase))
        {
            var parts = windowsPath.TrimStart('\\')
                                   .Split('\\', StringSplitOptions.RemoveEmptyEntries);
            // parts[0] = "wsl.localhost", parts[1] = distro name, rest = the path.
            return "/" + string.Join('/', parts.Skip(2));
        }

        if (windowsPath.Length > 2 && windowsPath[1] == ':')
        {
            var drive = char.ToLowerInvariant(windowsPath[0]);
            return $"/mnt/{drive}/" + windowsPath[3..].Replace('\\', '/');
        }

        return windowsPath.Replace('\\', '/');
    }

    /// <summary>Send one request and await its reply.</summary>
    /// <remarks>
    /// Requests are serialised: the engine answers one line at a time, so two
    /// overlapping calls would read each other's replies.
    /// </remarks>
    public async Task<JsonElement> RequestAsync(string command, object payload,
                                                CancellationToken token = default)
    {
        await _turn.WaitAsync(token).ConfigureAwait(false);
        try
        {
            var id = Interlocked.Increment(ref _nextId);
            var request = JsonSerializer.Serialize(new { id, command, payload });

            await _process.StandardInput.WriteLineAsync(request).ConfigureAwait(false);
            await _process.StandardInput.FlushAsync(token).ConfigureAwait(false);

            var line = await _process.StandardOutput.ReadLineAsync(token)
                                     .ConfigureAwait(false);
            if (line is null)
            {
                var errors = await _process.StandardError.ReadToEndAsync(token)
                                           .ConfigureAwait(false);
                throw new IOException(
                    "The Python engine closed unexpectedly. " + errors.Trim());
            }

            using var document = JsonDocument.Parse(line);
            var root = document.RootElement;
            if (!root.GetProperty("ok").GetBoolean())
            {
                throw new InvalidOperationException(
                    root.TryGetProperty("error", out var message)
                        ? message.GetString()
                        : "unknown engine error");
            }

            // The JsonDocument owns its buffer, so hand back a detached copy.
            return root.GetProperty("result").Clone();
        }
        finally
        {
            _turn.Release();
        }
    }

    public void Dispose()
    {
        try
        {
            if (!_process.HasExited)
            {
                _process.StandardInput.Close();
                if (!_process.WaitForExit(2000))
                {
                    _process.Kill(entireProcessTree: true);
                }
            }
        }
        catch
        {
            // Shutting down; nothing useful to do if the child is already gone.
        }

        _process.Dispose();
        _turn.Dispose();
    }
}
