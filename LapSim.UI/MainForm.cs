using System.Text;
using System.Text.Json;
using System.Windows.Forms;

namespace LapSim;

/// <summary>
/// The application window: controls on the left, results on the right.
/// </summary>
/// <remarks>
/// Built in code rather than with the designer, so the layout is reviewable as
/// ordinary source instead of generated XML.
///
/// Everything that is cheap re-runs as you type: a single simulation takes about
/// fifteen milliseconds, so the readouts follow the sliders. The searches are not
/// cheap -- they evaluate a thousand mixes or more -- so those sit behind buttons
/// and run off the UI thread.
/// </remarks>
public sealed class MainForm : Form
{
    private PythonBridge? _bridge;

    // Materials
    private readonly ComboBox _abrasive = new();
    private readonly ComboBox _grit = new();
    private readonly ComboBox _stone = new();
    private readonly ComboBox _lap = new();

    // The slurry itself
    private readonly TrackBar _solidsSlider = new();
    private readonly NumericUpDown _solids = new();
    private readonly TrackBar _glycerineSlider = new();
    private readonly NumericUpDown _glycerine = new();
    private readonly NumericUpDown _soapDrops = new();

    // How it is worked
    private readonly NumericUpDown _load = new();
    private readonly NumericUpDown _area = new();
    private readonly NumericUpDown _speed = new();
    private readonly NumericUpDown _faceLength = new();

    // The room and the plate
    private readonly NumericUpDown _temperature = new();
    private readonly NumericUpDown _humidity = new();
    private readonly NumericUpDown _airSpeed = new();
    private readonly NumericUpDown _filmVolume = new();

    private readonly Button _findRecipe = new();
    private readonly Button _mapWindow = new();
    private readonly Button _planSequence = new();

    private readonly TabControl _tabs = new();
    private readonly ListView _glance = new();
    private readonly TextBox _report = new();
    private readonly TextBox _recipeText = new();
    private readonly ListView _margins = new();
    private readonly WindowMapControl _map = new();
    private readonly ListView _plan = new();
    private readonly TextBox _planNotes = new();
    private readonly PictureBox _figure = new();
    private readonly ComboBox _figureChoice = new();
    private readonly Button _renderFigures = new();

    private readonly StatusStrip _status = new();
    private readonly ToolStripStatusLabel _statusText = new();
    private readonly ToolStripStatusLabel _engineText = new();
    private readonly System.Windows.Forms.Timer _debounce = new();

    private bool _loading = true;
    private int _busy;
    private JsonElement _capabilities;
    private string _repositoryRoot = "";

    public MainForm()
    {
        Text = "Lapping slurry simulator";
        MinimumSize = new Size(1120, 720);
        Size = new Size(1360, 860);
        StartPosition = FormStartPosition.CenterScreen;
        Font = new Font("Segoe UI", 9f);

        BuildLayout();
        Load += OnFormLoad;
        FormClosed += (_, _) => _bridge?.Dispose();
    }

    // ----------------------------------------------------------------- layout

    private void BuildLayout()
    {
        var split = new SplitContainer
        {
            Dock = DockStyle.Fill,
            FixedPanel = FixedPanel.Panel1,
            SplitterDistance = 340,
            SplitterWidth = 6,
        };

        var controls = new FlowLayoutPanel
        {
            Dock = DockStyle.Fill,
            FlowDirection = FlowDirection.TopDown,
            WrapContents = false,
            AutoScroll = true,
            Padding = new Padding(8),
        };
        controls.Controls.Add(BuildMaterialsGroup());
        controls.Controls.Add(BuildSlurryGroup());
        controls.Controls.Add(BuildProcessGroup());
        controls.Controls.Add(BuildRoomGroup());
        controls.Controls.Add(BuildActionsPanel());
        split.Panel1.Controls.Add(controls);

        BuildTabs();
        split.Panel2.Controls.Add(_tabs);

        _statusText.Spring = true;
        _statusText.TextAlign = ContentAlignment.MiddleLeft;
        _statusText.Text = "Starting the engine…";
        _status.Items.Add(_statusText);
        _status.Items.Add(_engineText);

        Controls.Add(split);
        Controls.Add(BuildMenu());
        Controls.Add(_status);

        _debounce.Interval = 150;
        _debounce.Tick += async (_, _) =>
        {
            _debounce.Stop();
            await SimulateAsync();
        };
    }

    private MenuStrip BuildMenu()
    {
        var file = new ToolStripMenuItem("&File");
        file.DropDownItems.Add("&Save report…", null, (_, _) => SaveReport());
        file.DropDownItems.Add(new ToolStripSeparator());
        file.DropDownItems.Add("E&xit", null, (_, _) => Close());

        var tools = new ToolStripMenuItem("&Tools");
        tools.DropDownItems.Add("&Reset to defaults", null, (_, _) => ApplyDefaults());
        tools.DropDownItems.Add("Open &figures folder", null, (_, _) => OpenFigures());

        var help = new ToolStripMenuItem("&Help");
        help.DropDownItems.Add("&About", null, (_, _) => MessageBox.Show(this,
            "Lapping slurry simulator\n\n"
            + "A physics model of loose-abrasive lapping between two flat plates. "
            + "The mechanics run in Python; this window drives them.\n\n"
            + "Relative comparisons follow from the physics. Absolute numbers rest "
            + "on fitted constants in lapsim/calibration.py — retune them "
            + "against your own bench measurements.",
            "About", MessageBoxButtons.OK, MessageBoxIcon.Information));

        var menu = new MenuStrip();
        menu.Items.AddRange([file, tools, help]);
        return menu;
    }

    /// <summary>Wrap a control in a fixed-size group box.</summary>
    /// <remarks>
    /// The height is given rather than derived: a docked, auto-sizing child
    /// inside an auto-sizing group box settles at whichever of the two wins,
    /// which is not something to leave to chance in a column of five groups.
    /// </remarks>
    private static GroupBox Group(string title, Control inner, int contentHeight)
    {
        inner.Dock = DockStyle.Fill;
        return new GroupBox
        {
            Text = title,
            Width = 310,
            Height = contentHeight + 26,
            Padding = new Padding(8, 4, 8, 8),
            Controls = { inner },
        };
    }

    /// <summary>A label above a control, stacked into a fixed-width column.</summary>
    private static void AddRow(TableLayoutPanel table, string label, Control control)
    {
        var row = table.RowCount++;
        table.RowStyles.Add(new RowStyle(SizeType.Absolute, 26));
        table.Controls.Add(new Label
        {
            Text = label,
            TextAlign = ContentAlignment.MiddleLeft,
            Dock = DockStyle.Fill,
            Margin = new Padding(0),
        }, 0, row);
        control.Dock = DockStyle.Fill;
        control.Margin = new Padding(0, 2, 0, 2);
        table.Controls.Add(control, 1, row);
    }

    private static TableLayoutPanel NewTable(int labelWidth = 122)
    {
        var table = new TableLayoutPanel
        {
            ColumnCount = 2,
            Dock = DockStyle.Fill,
            AutoSize = false,
        };
        table.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, labelWidth));
        table.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        return table;
    }

    private Control BuildMaterialsGroup()
    {
        var table = NewTable();
        foreach (var box in new[] { _abrasive, _grit, _stone, _lap })
        {
            box.DropDownStyle = ComboBoxStyle.DropDownList;
        }

        AddRow(table, "Abrasive", _abrasive);
        AddRow(table, "Grade", _grit);
        AddRow(table, "Stone", _stone);
        AddRow(table, "Lap plate", _lap);

        _abrasive.SelectedIndexChanged += (_, _) => { PopulateGrades(); QueueSimulate(); };
        _grit.SelectedIndexChanged += (_, _) => QueueSimulate();
        _stone.SelectedIndexChanged += (_, _) => QueueSimulate();
        _lap.SelectedIndexChanged += (_, _) => QueueSimulate();

        return Group("Abrasive and materials", table, 26 * 4 + 6);
    }

    private Control BuildSlurryGroup()
    {
        var table = NewTable();
        ConfigureNumeric(_solids, 2, 90, 1, 0, 40);
        ConfigureNumeric(_glycerine, 0, 95, 1, 0, 60);
        ConfigureNumeric(_soapDrops, 0, 20, 1, 0, 2);
        ConfigureSlider(_solidsSlider, 2, 90, 40);
        ConfigureSlider(_glycerineSlider, 0, 95, 60);

        Link(_solidsSlider, _solids);
        Link(_glycerineSlider, _glycerine);

        AddRow(table, "Grit loading (wt %)", _solids);
        table.RowCount++;
        table.RowStyles.Add(new RowStyle(SizeType.Absolute, 34));
        table.Controls.Add(_solidsSlider, 1, table.RowCount - 1);
        _solidsSlider.Dock = DockStyle.Fill;

        AddRow(table, "Glycerine (wt %)", _glycerine);
        table.RowCount++;
        table.RowStyles.Add(new RowStyle(SizeType.Absolute, 34));
        table.Controls.Add(_glycerineSlider, 1, table.RowCount - 1);
        _glycerineSlider.Dock = DockStyle.Fill;

        AddRow(table, "Dish soap (drops)", _soapDrops);
        _soapDrops.ValueChanged += (_, _) => QueueSimulate();

        return Group("Slurry  —  per 100 ml", table, 26 * 3 + 34 * 2 + 6);
    }

    private Control BuildProcessGroup()
    {
        var table = NewTable();
        ConfigureNumeric(_load, 2, 400, 1, 0, 40);
        ConfigureNumeric(_area, 20, 5000, 10, 0, 400);
        ConfigureNumeric(_speed, 0.02m, 2.0m, 0.01m, 2, 0.25m);
        ConfigureNumeric(_faceLength, 3, 200, 1, 0, 20);

        AddRow(table, "Pressing force (N)", _load);
        AddRow(table, "Face area (mm²)", _area);
        AddRow(table, "Stroke speed (m/s)", _speed);
        AddRow(table, "Face length (mm)", _faceLength);

        return Group("How you work", table, 26 * 4 + 6);
    }

    private Control BuildRoomGroup()
    {
        var table = NewTable();
        ConfigureNumeric(_temperature, 5, 40, 1, 0, 21);
        ConfigureNumeric(_humidity, 5, 95, 1, 0, 45);
        ConfigureNumeric(_airSpeed, 0, 3, 0.1m, 1, 0.3m);
        ConfigureNumeric(_filmVolume, 1, 60, 1, 0, 8);

        AddRow(table, "Temperature (°C)", _temperature);
        AddRow(table, "Humidity (% RH)", _humidity);
        AddRow(table, "Air movement (m/s)", _airSpeed);
        AddRow(table, "Slurry on plate (ml)", _filmVolume);

        return Group("The room and the plate", table, 26 * 4 + 6);
    }

    private Control BuildActionsPanel()
    {
        var panel = new FlowLayoutPanel
        {
            FlowDirection = FlowDirection.TopDown,
            WrapContents = false,
            Width = 310,
            Height = 112,
            Margin = new Padding(0, 6, 0, 0),
        };

        foreach (var (button, text, handler) in new (Button, string, EventHandler)[]
                 {
                     (_findRecipe, "Find the best recipe for this grade",
                      async (_, _) => await FindRecipeAsync()),
                     (_mapWindow, "Map the window of workable mixes",
                      async (_, _) => await MapWindowAsync()),
                     (_planSequence, "Plan the whole grit sequence",
                      async (_, _) => await PlanAsync()),
                 })
        {
            button.Text = text;
            button.Width = 296;
            button.Height = 30;
            button.Margin = new Padding(0, 0, 0, 5);
            button.Click += handler;
            panel.Controls.Add(button);
        }

        return panel;
    }

    private void BuildTabs()
    {
        _tabs.Dock = DockStyle.Fill;

        ConfigureReadout(_report);
        ConfigureReadout(_recipeText);
        ConfigureReadout(_planNotes);

        // At a glance
        ConfigureList(_glance, ("Quantity", 260), ("Value", 220));
        _tabs.TabPages.Add(NewPage("At a glance", _glance));

        // Full report
        _tabs.TabPages.Add(NewPage("Full report", _report));

        // Recipe search
        var recipeSplit = new SplitContainer
        {
            Dock = DockStyle.Fill,
            Orientation = Orientation.Horizontal,
            SplitterDistance = 200,
        };
        ConfigureList(_margins, ("Limit", 300), ("Margin", 100), ("Reading", 260));
        recipeSplit.Panel1.Controls.Add(_recipeText);
        recipeSplit.Panel2.Controls.Add(_margins);
        _tabs.TabPages.Add(NewPage("Best recipe", recipeSplit));

        // The window map
        _map.Dock = DockStyle.Fill;
        _map.PointPicked += (_, e) =>
        {
            SetValue(_solids, (decimal)Math.Round(e.SolidsWtPct));
            SetValue(_glycerine, (decimal)Math.Round(e.GlycerineWtPct));
            _solidsSlider.Value = (int)_solids.Value;
            _glycerineSlider.Value = (int)_glycerine.Value;
            QueueSimulate();
        };
        _tabs.TabPages.Add(NewPage("Recipe window", _map));

        // The plan
        var planSplit = new SplitContainer
        {
            Dock = DockStyle.Fill,
            Orientation = Orientation.Horizontal,
            SplitterDistance = 300,
        };
        ConfigureList(_plan,
            ("Abrasive", 170), ("Grade", 60), ("d50 µm", 65),
            ("Ra µm", 65), ("Damage µm", 85), ("Cut µm/min", 85),
            ("Clear µm", 80), ("Minutes", 65), ("Recipe per 100 ml", 340));
        planSplit.Panel1.Controls.Add(_plan);
        planSplit.Panel2.Controls.Add(_planNotes);
        _tabs.TabPages.Add(NewPage("Grit sequence", planSplit));

        // Rendered figures
        var figurePanel = new Panel { Dock = DockStyle.Fill };
        var figureBar = new FlowLayoutPanel
        {
            Dock = DockStyle.Top,
            Height = 34,
            Padding = new Padding(4),
        };
        _figureChoice.DropDownStyle = ComboBoxStyle.DropDownList;
        _figureChoice.Width = 300;
        _figureChoice.SelectedIndexChanged += (_, _) => ShowChosenFigure();
        _renderFigures.Text = "Render figures";
        _renderFigures.Width = 130;
        _renderFigures.Click += async (_, _) => await RenderFiguresAsync();
        figureBar.Controls.Add(_renderFigures);
        figureBar.Controls.Add(_figureChoice);

        _figure.Dock = DockStyle.Fill;
        _figure.SizeMode = PictureBoxSizeMode.Zoom;
        _figure.BackColor = Color.FromArgb(0xF9, 0xF9, 0xF7);
        figurePanel.Controls.Add(_figure);
        figurePanel.Controls.Add(figureBar);
        _tabs.TabPages.Add(NewPage("Figures", figurePanel));
    }

    private static TabPage NewPage(string title, Control content)
    {
        content.Dock = DockStyle.Fill;
        var page = new TabPage(title) { Padding = new Padding(6) };
        page.Controls.Add(content);
        return page;
    }

    private static void ConfigureReadout(TextBox box)
    {
        box.Multiline = true;
        box.ReadOnly = true;
        box.ScrollBars = ScrollBars.Both;
        box.WordWrap = false;
        box.Font = new Font("Consolas", 9f);
        box.BackColor = Color.White;
    }

    private static void ConfigureList(ListView list, params (string Header, int Width)[] columns)
    {
        list.View = View.Details;
        list.FullRowSelect = true;
        list.GridLines = false;
        list.HideSelection = false;
        foreach (var (header, width) in columns)
        {
            list.Columns.Add(header, width);
        }
    }

    private static void ConfigureNumeric(NumericUpDown box, decimal minimum,
                                         decimal maximum, decimal step,
                                         int decimals, decimal value)
    {
        box.Minimum = minimum;
        box.Maximum = maximum;
        box.Increment = step;
        box.DecimalPlaces = decimals;
        box.Value = value;
    }

    private static void ConfigureSlider(TrackBar slider, int minimum, int maximum,
                                        int value)
    {
        slider.Minimum = minimum;
        slider.Maximum = maximum;
        slider.Value = value;
        slider.TickFrequency = 10;
        slider.TickStyle = TickStyle.BottomRight;
    }

    /// <summary>Keep a slider and its spin box showing the same number.</summary>
    private void Link(TrackBar slider, NumericUpDown box)
    {
        slider.ValueChanged += (_, _) =>
        {
            if (box.Value != slider.Value)
            {
                box.Value = slider.Value;
            }
            QueueSimulate();
        };
        box.ValueChanged += (_, _) =>
        {
            var clamped = Math.Clamp((int)box.Value, slider.Minimum, slider.Maximum);
            if (slider.Value != clamped)
            {
                slider.Value = clamped;
            }
            QueueSimulate();
        };
    }

    // ------------------------------------------------------------- start-up

    private async void OnFormLoad(object? sender, EventArgs e)
    {
        _repositoryRoot = FindRepositoryRoot();
        try
        {
            _bridge = await Task.Run(() => PythonBridge.Start(_repositoryRoot));
        }
        catch (Exception error)
        {
            MessageBox.Show(this, error.Message, "Cannot start the engine",
                            MessageBoxButtons.OK, MessageBoxIcon.Error);
            _statusText.Text = "No engine. Set LAPSIM_PYTHON and restart.";
            return;
        }

        _capabilities = _bridge.Capabilities;
        _engineText.Text = $"Python {_capabilities.GetProperty("python").GetString()} "
                           + $"via {_bridge.Description}";
        _renderFigures.Enabled = _bridge.CanPlot;
        _renderFigures.Text = _bridge.CanPlot
            ? "Render figures"
            : "Render (needs matplotlib)";
        LoadExistingFigures();

        PopulateMaterials();
        _loading = false;
        await SimulateAsync();
    }

    /// <summary>Walk up from the executable until the <c>lapsim</c> package appears.</summary>
    private static string FindRepositoryRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "lapsim", "simulate.py")))
            {
                return directory.FullName;
            }
            directory = directory.Parent;
        }

        return Directory.GetCurrentDirectory();
    }

    private void PopulateMaterials()
    {
        _abrasive.Items.Clear();
        foreach (var abrasive in _capabilities.GetProperty("abrasives").EnumerateArray())
        {
            _abrasive.Items.Add(new Entry(abrasive.GetProperty("key").GetString()!,
                                          abrasive.GetProperty("name").GetString()!));
        }
        _abrasive.SelectedIndex = 0;

        _stone.Items.Clear();
        foreach (var stone in _capabilities.GetProperty("stones").EnumerateArray())
        {
            _stone.Items.Add(new Entry(stone.GetProperty("key").GetString()!,
                                       stone.GetProperty("name").GetString()!));
        }
        _stone.SelectedIndex = 0;

        _lap.Items.Clear();
        foreach (var lap in _capabilities.GetProperty("laps").EnumerateArray())
        {
            _lap.Items.Add(new Entry(lap.GetProperty("key").GetString()!,
                                     lap.GetProperty("name").GetString()!));
        }
        _lap.SelectedIndex = 0;

        PopulateGrades();
        ApplyDefaults();
    }

    /// <summary>List the grades of the chosen abrasive, marking what is in stock.</summary>
    private void PopulateGrades()
    {
        if (_abrasive.SelectedItem is not Entry chosen)
        {
            return;
        }

        var wasLoading = _loading;
        _loading = true;
        _grit.Items.Clear();

        var preferred = 0;
        foreach (var abrasive in _capabilities.GetProperty("abrasives").EnumerateArray())
        {
            if (abrasive.GetProperty("key").GetString() != chosen.Key)
            {
                continue;
            }

            foreach (var grade in abrasive.GetProperty("grades").EnumerateArray())
            {
                var label = grade.GetProperty("label").GetString()!;
                var inStock = grade.GetProperty("in_stock").GetBoolean();
                var d50 = grade.GetProperty("d50_um").GetDouble();
                var text = $"{label}  —  d50 {d50:0.#} µm"
                           + (inStock ? "  (in stock)" : "");
                if (inStock && preferred == 0)
                {
                    preferred = _grit.Items.Count;
                }
                _grit.Items.Add(new Entry(label, text));
            }
        }

        if (_grit.Items.Count > 0)
        {
            _grit.SelectedIndex = Math.Min(preferred, _grit.Items.Count - 1);
        }

        _loading = wasLoading;
    }

    private void ApplyDefaults()
    {
        var defaults = _capabilities.GetProperty("defaults");
        var wasLoading = _loading;
        _loading = true;

        SetValue(_solids, (decimal)(defaults.GetProperty("solids_mass_fraction")
                                            .GetDouble() * 100));
        SetValue(_glycerine, (decimal)(defaults.GetProperty("glycerine_mass_fraction")
                                               .GetDouble() * 100));
        SetValue(_soapDrops, 2);
        SetValue(_load, (decimal)defaults.GetProperty("normal_load_N").GetDouble());
        SetValue(_area, (decimal)defaults.GetProperty("contact_area_mm2").GetDouble());
        SetValue(_speed, (decimal)defaults.GetProperty("sliding_speed_m_s").GetDouble());
        SetValue(_faceLength,
                 (decimal)defaults.GetProperty("contact_length_mm").GetDouble());
        SetValue(_temperature,
                 (decimal)defaults.GetProperty("temperature_C").GetDouble());
        SetValue(_humidity,
                 (decimal)(defaults.GetProperty("relative_humidity").GetDouble() * 100));
        SetValue(_airSpeed, (decimal)defaults.GetProperty("air_speed_m_s").GetDouble());
        SetValue(_filmVolume, (decimal)defaults.GetProperty("film_volume_ml").GetDouble());

        _solidsSlider.Value = (int)_solids.Value;
        _glycerineSlider.Value = (int)_glycerine.Value;

        _loading = wasLoading;
        QueueSimulate();
    }

    private static void SetValue(NumericUpDown box, decimal value) =>
        box.Value = Math.Clamp(value, box.Minimum, box.Maximum);

    private sealed record Entry(string Key, string Text)
    {
        public override string ToString() => Text;
    }

    // ---------------------------------------------------------- the requests

    private object BuildPayload()
    {
        // One drop is 0.05 ml, and the payload wants a fraction of the liquid.
        var soapFraction = (double)_soapDrops.Value * 0.05 / 100.0;
        return new
        {
            abrasive = ((Entry)_abrasive.SelectedItem!).Key,
            grit = ((Entry)_grit.SelectedItem!).Key,
            stone = ((Entry)_stone.SelectedItem!).Key,
            lap = ((Entry)_lap.SelectedItem!).Key,
            solids_mass_fraction = (double)_solids.Value / 100.0,
            glycerine_mass_fraction = (double)_glycerine.Value / 100.0,
            soap_volume_fraction = Math.Max(soapFraction, 1e-6),
            normal_load_N = (double)_load.Value,
            contact_area_mm2 = (double)_area.Value,
            sliding_speed_m_s = (double)_speed.Value,
            contact_length_mm = (double)_faceLength.Value,
            temperature_C = (double)_temperature.Value,
            relative_humidity = (double)_humidity.Value / 100.0,
            air_speed_m_s = (double)_airSpeed.Value,
            film_volume_ml = (double)_filmVolume.Value,
        };
    }

    private void QueueSimulate()
    {
        if (_loading || _bridge is null)
        {
            return;
        }

        _debounce.Stop();
        _debounce.Start();
    }

    /// <summary>Run a slow request with the controls disabled and a status message.</summary>
    private async Task WithBusy(string message, Func<Task> work)
    {
        if (_bridge is null || Interlocked.Exchange(ref _busy, 1) == 1)
        {
            return;
        }

        _findRecipe.Enabled = _mapWindow.Enabled = _planSequence.Enabled = false;
        _statusText.Text = message;
        UseWaitCursor = true;
        try
        {
            await work();
        }
        catch (Exception error)
        {
            _statusText.Text = "Failed: " + error.Message;
            MessageBox.Show(this, error.Message, "The engine reported a problem",
                            MessageBoxButtons.OK, MessageBoxIcon.Warning);
        }
        finally
        {
            UseWaitCursor = false;
            _findRecipe.Enabled = _mapWindow.Enabled = _planSequence.Enabled = true;
            Interlocked.Exchange(ref _busy, 0);
        }
    }

    private async Task SimulateAsync()
    {
        if (_bridge is null || _grit.SelectedItem is null)
        {
            return;
        }

        try
        {
            var result = await _bridge.RequestAsync("simulate", BuildPayload());
            ShowGlance(result);
            _report.Text = result.GetProperty("report").GetString()!
                                 .Replace("\n", Environment.NewLine);

            var warnings = result.GetProperty("warnings").EnumerateArray().ToList();
            _statusText.Text = warnings.Count == 0
                ? "This mix sits inside every target band."
                : $"{warnings.Count} warning(s): {warnings[0].GetString()}";
        }
        catch (Exception error)
        {
            _statusText.Text = "Failed: " + error.Message;
        }
    }

    private void ShowGlance(JsonElement result)
    {
        var slurry = result.GetProperty("slurry");
        var contact = result.GetProperty("contact");
        var cutting = result.GetProperty("cutting");
        var finish = result.GetProperty("finish");
        var drying = result.GetProperty("drying");

        _glance.BeginUpdate();
        _glance.Items.Clear();
        _glance.Groups.Clear();

        void Section(string title, params (string Label, string Value)[] rows)
        {
            var group = new ListViewGroup(title);
            _glance.Groups.Add(group);
            foreach (var (label, value) in rows)
            {
                _glance.Items.Add(new ListViewItem([label, value]) { Group = group });
            }
        }

        Section("Cutting",
            ("Stone removal", $"{cutting.GetProperty("removal_um_min").GetDouble():0.0} µm/min"),
            ("Plate wear", $"{cutting.GetProperty("lap_wear_um_min").GetDouble():0.0} µm/min "
                           + $"({cutting.GetProperty("plate_to_stone_wear").GetDouble():0.00}× the stone)"),
            ("Drag you feel", $"{cutting.GetProperty("drag_N").GetDouble():0.0} N "
                              + $"({cutting.GetProperty("drag_ratio").GetDouble():0.00}× the load)"),
            ("Contact warms by", $"{cutting.GetProperty("temperature_rise_K").GetDouble():0} K"));

        Section("Finish",
            ("Roughness Ra", $"{finish.GetProperty("Ra_um").GetDouble():0.00} µm"),
            ("Peak to valley Rz", $"{finish.GetProperty("Rz_um").GetDouble():0.00} µm"),
            ("Damage below the surface",
             $"{finish.GetProperty("subsurface_damage_um").GetDouble():0.0} µm "
             + "— the next grade must remove this much"),
            ("Worst-case scratch",
             $"{finish.GetProperty("deepest_scratch_um").GetDouble():0.00} µm"),
            ("Cutting regime", finish.GetProperty("regime").GetString()!));

        Section("The slurry",
            ("Consistency", slurry.GetProperty("consistency").GetString()!),
            ("Viscosity", $"{slurry.GetProperty("slurry_viscosity_mPas").GetDouble():0} mPa·s "
                          + "while cutting"),
            ("Yield stress", $"{slurry.GetProperty("yield_stress_Pa").GetDouble():0.0} Pa"),
            ("Blob height on the plate",
             $"{slurry.GetProperty("puddle_height_mm").GetDouble():0.00} mm  — check this by eye"),
            ("Grit stays suspended", Duration(slurry, "settling_time_s")),
            ("Solids", $"{slurry.GetProperty("solids_wt_pct").GetDouble():0} wt% "
                       + $"= {slurry.GetProperty("solids_vol_pct").GetDouble():0.0} vol% "
                       + $"(jams at {slurry.GetProperty("jamming_vol_pct").GetDouble():0} vol%)"));

        Section("In the gap",
            ("Regime", contact.GetProperty("regime").GetString()!),
            ("Plate gap", $"{contact.GetProperty("gap_um").GetDouble():0.0} µm "
                          + $"= {contact.GetProperty("gap_over_d50").GetDouble():0.0}× d50"),
            ("Load carried by grains",
             $"{contact.GetProperty("load_fraction_on_grains").GetDouble() * 100:0} %"),
            ("Grains actually cutting",
             $"{contact.GetProperty("active_grains_per_mm2").GetDouble():0.0} per mm² "
             + $"({contact.GetProperty("active_fraction_pct").GetDouble():0.000} % of those present)"),
            ("Grains being crushed",
             $"{contact.GetProperty("fractured_pct").GetDouble():0.0} %"));

        Section("Working life",
            ("Film on the plate",
             $"{drying.GetProperty("film_thickness_um").GetDouble():0} µm thick, "
             + $"losing {drying.GetProperty("evaporation_g_min").GetDouble():0.00} g water/min"),
            ("Stays usable for", Duration(drying, "workable_time_s")),
            ("Dries down to",
             $"{drying.GetProperty("equilibrium_solids_pct").GetDouble():0} vol% solids "
             + $"(unworkable above {drying.GetProperty("jamming_solids_pct").GetDouble():0} vol%)"),
            ("Can it dry to a crust?",
             drying.GetProperty("seizes").GetBoolean() ? "Yes — it will" : "No, never"));

        _glance.EndUpdate();
    }

    private static string Duration(JsonElement parent, string property)
    {
        var value = parent.GetProperty(property);
        if (value.ValueKind == JsonValueKind.Null)
        {
            return "indefinitely";
        }

        var seconds = value.GetDouble();
        return seconds >= 60 ? $"{seconds / 60:0} min" : $"{seconds:0} s";
    }

    private async Task FindRecipeAsync() => await WithBusy(
        "Searching every mix for this grade…", async () =>
        {
            var payload = BuildPayload();
            var result = await _bridge!.RequestAsync("recipe", payload);

            if (!result.GetProperty("found").GetBoolean())
            {
                var explanation = result.GetProperty("explanation").GetString()!;
                _recipeText.Text = Wrap(explanation, 96);
                _margins.Items.Clear();
                _tabs.SelectedIndex = 2;
                _statusText.Text = "No mix works here — see the suggestions.";
                return;
            }

            var recipe = result.GetProperty("recipe");
            var text = new StringBuilder();
            text.AppendLine(recipe.GetProperty("text").GetString()!
                                  .Replace("\n", Environment.NewLine));
            text.AppendLine();
            text.AppendLine(result.GetProperty("tolerance_text").GetString());
            text.AppendLine();
            text.AppendLine($"Tightest limit: {result.GetProperty("binding_constraint").GetString()} "
                            + $"(margin {result.GetProperty("robustness").GetDouble():+0.00;-0.00})");
            _recipeText.Text = text.ToString();

            _margins.BeginUpdate();
            _margins.Items.Clear();
            foreach (var margin in result.GetProperty("margins").EnumerateObject())
            {
                var value = margin.Value.GetDouble();
                var reading = value switch
                {
                    < 0 => "breaks this limit",
                    < 0.15 => "only just inside — fragile",
                    < 0.5 => "comfortable",
                    _ => "plenty of room",
                };
                _margins.Items.Add(new ListViewItem(
                    [margin.Name, value.ToString("+0.00;-0.00"), reading]));
            }
            _margins.EndUpdate();

            // Load the winning mix into the controls so the readouts follow it.
            _loading = true;
            SetValue(_solids, (decimal)(result.GetProperty("solids_mass_fraction")
                                              .GetDouble() * 100));
            SetValue(_glycerine, (decimal)(result.GetProperty("glycerine_mass_fraction")
                                                 .GetDouble() * 100));
            SetValue(_soapDrops, recipe.GetProperty("soap_drops").GetInt32());
            _solidsSlider.Value = (int)_solids.Value;
            _glycerineSlider.Value = (int)_glycerine.Value;
            _map.MarkerSolidsWtPct = (double)_solids.Value;
            _map.MarkerGlycerineWtPct = (double)_glycerine.Value;
            _map.Invalidate();
            _loading = false;

            _tabs.SelectedIndex = 2;
            await SimulateAsync();
            _statusText.Text = "Best recipe loaded into the controls.";
        });

    private async Task MapWindowAsync() => await WithBusy(
        "Scanning the recipe window…", async () =>
        {
            var result = await _bridge!.RequestAsync("window",
                Merge(BuildPayload(), new { resolution = 30 }));

            var solids = result.GetProperty("solids_wt_pct")
                               .EnumerateArray().Select(v => v.GetDouble()).ToArray();
            var glycerine = result.GetProperty("glycerine_wt_pct")
                                  .EnumerateArray().Select(v => v.GetDouble()).ToArray();

            _map.SetData(solids, glycerine,
                         ToGrid(result.GetProperty("robustness")),
                         ToGrid(result.GetProperty("removal_um_min")),
                         ToStringGrid(result.GetProperty("binding")),
                         $"{result.GetProperty("abrasive_name").GetString()} "
                         + $"{result.GetProperty("grit").GetString()}  —  "
                         + $"{result.GetProperty("feasible_fraction").GetDouble() * 100:0} % "
                         + "of this map works. Click to try a mix.");
            _map.MarkerSolidsWtPct = (double)_solids.Value;
            _map.MarkerGlycerineWtPct = (double)_glycerine.Value;
            _tabs.SelectedIndex = 3;
            _statusText.Text = "Hover to read any mix; click to load it.";
        });

    private async Task PlanAsync() => await WithBusy(
        "Planning the sequence — this searches every grade…", async () =>
        {
            var result = await _bridge!.RequestAsync("plan",
                Merge(BuildPayload(), new { resolution = 14 }));

            _plan.BeginUpdate();
            _plan.Items.Clear();
            foreach (var stage in result.GetProperty("stages").EnumerateArray())
            {
                var recipe = stage.GetProperty("recipe");
                var summary =
                    $"{recipe.GetProperty("grit_grams").GetDouble():0} g grit + "
                    + $"{recipe.GetProperty("water_ml").GetDouble():0} ml water + "
                    + $"{recipe.GetProperty("glycerine_ml").GetDouble():0} ml glycerine + "
                    + $"{recipe.GetProperty("soap_drops").GetInt32()} drops";
                var item = new ListViewItem([
                    stage.GetProperty("abrasive_name").GetString()!,
                    stage.GetProperty("grit").GetString()!,
                    $"{stage.GetProperty("d50_um").GetDouble():0.#}",
                    $"{stage.GetProperty("Ra_um").GetDouble():0.000}",
                    $"{stage.GetProperty("subsurface_damage_um").GetDouble():0.0}",
                    $"{stage.GetProperty("removal_um_min").GetDouble():0.0}",
                    $"{stage.GetProperty("clear_um").GetDouble():0.0}",
                    $"{stage.GetProperty("minutes").GetDouble():0.0}",
                    summary,
                ]);
                if (!stage.GetProperty("in_stock").GetBoolean())
                {
                    item.ForeColor = Color.FromArgb(0x52, 0x51, 0x4E);
                    item.SubItems[0].Text += "  (to buy)";
                }
                _plan.Items.Add(item);
            }
            _plan.EndUpdate();

            var notes = new StringBuilder();
            notes.AppendLine($"Total {result.GetProperty("total_minutes").GetDouble():0} "
                             + "minutes of continuous grinding at the stated pressure "
                             + "and speed.");
            notes.AppendLine("Real bench time is longer: strokes lift, pressure varies, "
                             + "and the plate needs re-charging.");
            foreach (var skip in result.GetProperty("skipped").EnumerateArray())
            {
                notes.AppendLine();
                notes.AppendLine($"SKIPPED  {skip.GetProperty("abrasive_name").GetString()} "
                                 + skip.GetProperty("grit").GetString());
                notes.AppendLine(Wrap(skip.GetProperty("explanation").GetString()!, 96));
            }
            _planNotes.Text = notes.ToString();

            _tabs.SelectedIndex = 4;
            _statusText.Text = "Sequence planned.";
        });

    private async Task RenderFiguresAsync() => await WithBusy(
        "Rendering figures — this takes a minute…", async () =>
        {
            var result = await _bridge!.RequestAsync("figures", BuildPayload());
            _figureChoice.Items.Clear();
            foreach (var path in result.GetProperty("paths").EnumerateArray())
            {
                _figureChoice.Items.Add(path.GetString()!);
            }

            if (_figureChoice.Items.Count > 0)
            {
                _figureChoice.SelectedIndex = 0;
            }
            _tabs.SelectedIndex = 5;
            _statusText.Text = $"{_figureChoice.Items.Count} figures rendered.";
        });

    /// <summary>
    /// List any figures already on disk, so the tab is useful straight away --
    /// and stays useful with an interpreter that has no matplotlib, which can
    /// still run the whole model but cannot draw.
    /// </summary>
    private void LoadExistingFigures()
    {
        var folder = Path.Combine(_repositoryRoot, "figures");
        if (!Directory.Exists(folder))
        {
            return;
        }

        _figureChoice.Items.Clear();
        foreach (var path in Directory.GetFiles(folder, "*.png").OrderBy(p => p))
        {
            _figureChoice.Items.Add(path);
        }

        if (_figureChoice.Items.Count > 0)
        {
            _figureChoice.SelectedIndex = 0;
        }
        else if (_bridge is not null && !_bridge.CanPlot)
        {
            _statusText.Text = "This interpreter has no matplotlib, "
                               + "so figures cannot be rendered here.";
        }
    }

    private void ShowChosenFigure()
    {
        if (_figureChoice.SelectedItem is not string path || !File.Exists(path))
        {
            return;
        }

        // Load through a copy so the file is not left locked on disk.
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read);
        using var loaded = Image.FromStream(stream);
        _figure.Image?.Dispose();
        _figure.Image = new Bitmap(loaded);
    }

    // ------------------------------------------------------------- helpers

    private static double[][] ToGrid(JsonElement element) => element.EnumerateArray()
        .Select(row => row.EnumerateArray().Select(v => v.GetDouble()).ToArray())
        .ToArray();

    private static string[][] ToStringGrid(JsonElement element) => element.EnumerateArray()
        .Select(row => row.EnumerateArray().Select(v => v.GetString() ?? "").ToArray())
        .ToArray();

    /// <summary>Combine the standard payload with extra, request-specific keys.</summary>
    private static Dictionary<string, object?> Merge(object payload, object extra)
    {
        var merged = new Dictionary<string, object?>();
        foreach (var source in new[] { payload, extra })
        {
            foreach (var property in source.GetType().GetProperties())
            {
                merged[property.Name] = property.GetValue(source);
            }
        }
        return merged;
    }

    private static string Wrap(string text, int width)
    {
        var lines = new List<string>();
        var line = new StringBuilder();
        foreach (var word in text.Split(' ', StringSplitOptions.RemoveEmptyEntries))
        {
            if (line.Length > 0 && line.Length + word.Length + 1 > width)
            {
                lines.Add(line.ToString());
                line.Clear();
            }
            if (line.Length > 0)
            {
                line.Append(' ');
            }
            line.Append(word);
        }

        if (line.Length > 0)
        {
            lines.Add(line.ToString());
        }

        return string.Join(Environment.NewLine, lines);
    }

    private void SaveReport()
    {
        using var dialog = new SaveFileDialog
        {
            Filter = "Text files (*.txt)|*.txt|All files (*.*)|*.*",
            FileName = "slurry-report.txt",
        };

        if (dialog.ShowDialog(this) == DialogResult.OK)
        {
            File.WriteAllText(dialog.FileName, _report.Text);
            _statusText.Text = "Saved " + dialog.FileName;
        }
    }

    private void OpenFigures()
    {
        var folder = Path.Combine(_repositoryRoot, "figures");
        if (!Directory.Exists(folder))
        {
            MessageBox.Show(this, "No figures have been rendered yet.", "Figures",
                            MessageBoxButtons.OK, MessageBoxIcon.Information);
            return;
        }

        System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo
        {
            FileName = folder,
            UseShellExecute = true,
        });
    }
}
