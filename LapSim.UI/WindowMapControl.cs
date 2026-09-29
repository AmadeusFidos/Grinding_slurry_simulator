using System.ComponentModel;
using System.Drawing.Drawing2D;
using System.Windows.Forms;

namespace LapSim;

/// <summary>Arguments for <see cref="WindowMapControl.PointPicked"/>.</summary>
public sealed class RecipePickedEventArgs(double solidsWtPct, double glycerineWtPct)
    : EventArgs
{
    public double SolidsWtPct { get; } = solidsWtPct;
    public double GlycerineWtPct { get; } = glycerineWtPct;
}

/// <summary>
/// Draws the recipe window: grit loading across, glycerine up, colour showing
/// how much margin a mix has to the nearest limit.
/// </summary>
/// <remarks>
/// Red is outside every target band, grey is exactly on the boundary and blue is
/// inside, which puts the edge of the usable region where the colour turns
/// neutral. Click anywhere to load that mix into the controls.
///
/// Drawn with GDI+ rather than by rendering an image in Python, so it stays
/// interactive and works even where matplotlib is not installed.
/// </remarks>
public sealed class WindowMapControl : Control
{
    private const int MarginLeft = 54;
    private const int MarginBottom = 44;
    private const int MarginTop = 14;
    private const int MarginRight = 16;

    private double[] _solidsAxis = [];
    private double[] _glycerineAxis = [];
    private double[][] _robustness = [];
    private double[][] _removalRate = [];
    private string[][] _binding = [];
    private Bitmap? _surface;
    private Point _cursor = Point.Empty;
    private bool _hovering;

    // These are driven from code, never from the designer, so they are kept out
    // of the designer's serialisation and property grid.
    [Browsable(false)]
    [DesignerSerializationVisibility(DesignerSerializationVisibility.Hidden)]
    public double MarkerSolidsWtPct { get; set; } = double.NaN;

    [Browsable(false)]
    [DesignerSerializationVisibility(DesignerSerializationVisibility.Hidden)]
    public double MarkerGlycerineWtPct { get; set; } = double.NaN;

    [Browsable(false)]
    [DesignerSerializationVisibility(DesignerSerializationVisibility.Hidden)]
    public string Caption { get; set; } = "";

    public event EventHandler<RecipePickedEventArgs>? PointPicked;

    public WindowMapControl()
    {
        DoubleBuffered = true;
        SetStyle(ControlStyles.ResizeRedraw | ControlStyles.OptimizedDoubleBuffer, true);
        BackColor = Color.FromArgb(0xFC, 0xFC, 0xFB);
    }

    public bool HasData => _robustness.Length > 0;

    public void SetData(double[] solidsAxis, double[] glycerineAxis,
                        double[][] robustness, double[][] removalRate,
                        string[][] binding, string caption)
    {
        _solidsAxis = solidsAxis;
        _glycerineAxis = glycerineAxis;
        _robustness = robustness;
        _removalRate = removalRate;
        _binding = binding;
        Caption = caption;

        _surface?.Dispose();
        _surface = BuildSurface();
        Invalidate();
    }

    /// <summary>
    /// Rasterise the grid one pixel per cell; the paint pass scales it up with
    /// interpolation, which is both faster and smoother than filling rectangles.
    /// </summary>
    private Bitmap? BuildSurface()
    {
        if (_robustness.Length == 0 || _robustness[0].Length == 0)
        {
            return null;
        }

        var rows = _robustness.Length;
        var columns = _robustness[0].Length;
        var bitmap = new Bitmap(columns, rows);
        for (var row = 0; row < rows; row++)
        {
            for (var column = 0; column < columns; column++)
            {
                // Glycerine increases upward on screen but forward in the array.
                bitmap.SetPixel(column, rows - 1 - row,
                                MarginColour(_robustness[row][column]));
            }
        }

        return bitmap;
    }

    /// <summary>
    /// Diverging red - grey - blue, clipped to +-1.
    /// </summary>
    /// <remarks>
    /// Clipping matters: a few wildly unusable mixes would otherwise set the
    /// scale and wash out the boundary region, which is the only part worth
    /// reading. The steps match the Python figures so the two agree.
    /// </remarks>
    private static Color MarginColour(double margin)
    {
        double[] stops = [-1.0, -0.5, -0.15, 0.0, 0.15, 0.5, 1.0];
        Color[] colours =
        [
            Color.FromArgb(0x7A, 0x1D, 0x1C),
            Color.FromArgb(0xE3, 0x49, 0x48),
            Color.FromArgb(0xF5, 0xAD, 0xAD),
            Color.FromArgb(0xF0, 0xEF, 0xEC),
            Color.FromArgb(0x9E, 0xC5, 0xF4),
            Color.FromArgb(0x39, 0x87, 0xE5),
            Color.FromArgb(0x10, 0x42, 0x81),
        ];

        var value = Math.Clamp(margin, -1.0, 1.0);
        for (var i = 0; i < stops.Length - 1; i++)
        {
            if (value <= stops[i + 1])
            {
                var span = stops[i + 1] - stops[i];
                var t = span <= 0 ? 0 : (value - stops[i]) / span;
                return Blend(colours[i], colours[i + 1], t);
            }
        }

        return colours[^1];
    }

    private static Color Blend(Color from, Color to, double t) => Color.FromArgb(
        (int)Math.Round(from.R + (to.R - from.R) * t),
        (int)Math.Round(from.G + (to.G - from.G) * t),
        (int)Math.Round(from.B + (to.B - from.B) * t));

    private Rectangle PlotArea => new(
        MarginLeft, MarginTop,
        Math.Max(Width - MarginLeft - MarginRight, 1),
        Math.Max(Height - MarginTop - MarginBottom, 1));

    protected override void OnPaint(PaintEventArgs e)
    {
        var graphics = e.Graphics;
        graphics.Clear(BackColor);
        graphics.TextRenderingHint = System.Drawing.Text.TextRenderingHint.ClearTypeGridFit;

        var plot = PlotArea;
        using var ink = new SolidBrush(Color.FromArgb(0x0B, 0x0B, 0x0B));
        using var muted = new SolidBrush(Color.FromArgb(0x52, 0x51, 0x4E));
        using var axisPen = new Pen(Color.FromArgb(0xC3, 0xC2, 0xB7));
        using var font = new Font(Font.FontFamily, 7.5f);

        if (_surface is null)
        {
            TextRenderer.DrawText(graphics,
                "Press “Map the window” to scan every mix for this grade.",
                Font, plot, Color.FromArgb(0x52, 0x51, 0x4E),
                TextFormatFlags.HorizontalCenter | TextFormatFlags.VerticalCenter);
            return;
        }

        graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
        graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;
        graphics.DrawImage(_surface, plot);
        graphics.DrawRectangle(axisPen, plot);

        DrawAxes(graphics, plot, muted, axisPen, font);
        DrawMarker(graphics, plot, ink);
        DrawReadout(graphics, plot, ink, font);
    }

    private void DrawAxes(Graphics graphics, Rectangle plot, Brush muted,
                          Pen axisPen, Font font)
    {
        if (_solidsAxis.Length == 0 || _glycerineAxis.Length == 0)
        {
            return;
        }

        for (var i = 0; i <= 5; i++)
        {
            var fraction = i / 5.0;

            var x = plot.Left + (float)(fraction * plot.Width);
            var solids = _solidsAxis[0] + fraction * (_solidsAxis[^1] - _solidsAxis[0]);
            graphics.DrawLine(axisPen, x, plot.Bottom, x, plot.Bottom + 4);
            var label = $"{solids:0}";
            var size = graphics.MeasureString(label, font);
            graphics.DrawString(label, font, muted, x - size.Width / 2, plot.Bottom + 5);

            var y = plot.Bottom - (float)(fraction * plot.Height);
            var glycerine = _glycerineAxis[0]
                            + fraction * (_glycerineAxis[^1] - _glycerineAxis[0]);
            graphics.DrawLine(axisPen, plot.Left - 4, y, plot.Left, y);
            label = $"{glycerine:0}";
            size = graphics.MeasureString(label, font);
            graphics.DrawString(label, font, muted,
                                plot.Left - 6 - size.Width, y - size.Height / 2);
        }

        using var axisFont = new Font(Font.FontFamily, 8f);
        var caption = "grit loading (wt % of the slurry)";
        var captionSize = graphics.MeasureString(caption, axisFont);
        graphics.DrawString(caption, axisFont, muted,
                            plot.Left + (plot.Width - captionSize.Width) / 2,
                            plot.Bottom + 20);

        var state = graphics.Save();
        graphics.TranslateTransform(12, plot.Top + plot.Height / 2f);
        graphics.RotateTransform(-90);
        var side = "glycerine in the liquid (wt %)";
        var sideSize = graphics.MeasureString(side, axisFont);
        graphics.DrawString(side, axisFont, muted, -sideSize.Width / 2, 0);
        graphics.Restore(state);
    }

    private void DrawMarker(Graphics graphics, Rectangle plot, Brush ink)
    {
        if (double.IsNaN(MarkerSolidsWtPct) || double.IsNaN(MarkerGlycerineWtPct)
            || _solidsAxis.Length == 0)
        {
            return;
        }

        var point = ValueToPoint(plot, MarkerSolidsWtPct, MarkerGlycerineWtPct);
        using var halo = new Pen(Color.White, 3f);
        using var pen = new Pen(Color.FromArgb(0x0B, 0x0B, 0x0B), 1.6f);
        graphics.DrawEllipse(halo, point.X - 6, point.Y - 6, 12, 12);
        graphics.DrawEllipse(pen, point.X - 6, point.Y - 6, 12, 12);
        graphics.FillEllipse(ink, point.X - 2, point.Y - 2, 4, 4);
    }

    private void DrawReadout(Graphics graphics, Rectangle plot, Brush ink, Font font)
    {
        var text = Caption;
        if (_hovering && plot.Contains(_cursor))
        {
            var (column, row) = PointToCell(plot, _cursor);
            var margin = _robustness[row][column];
            var verdict = margin > 0
                ? $"works, margin {margin:+0.00}"
                : $"fails on “{_binding[row][column]}”";
            text = $"{_solidsAxis[column]:0} wt% grit, "
                   + $"{_glycerineAxis[row]:0} % glycerine  —  {verdict}, "
                   + $"{_removalRate[row][column]:0.0} µm/min";
        }

        if (string.IsNullOrEmpty(text))
        {
            return;
        }

        var size = graphics.MeasureString(text, font);
        var box = new RectangleF(plot.Left + 4, plot.Top + 4,
                                 size.Width + 8, size.Height + 4);
        using var backing = new SolidBrush(Color.FromArgb(225, 252, 252, 251));
        graphics.FillRectangle(backing, box);
        graphics.DrawString(text, font, ink, box.X + 4, box.Y + 2);
    }

    private PointF ValueToPoint(Rectangle plot, double solids, double glycerine)
    {
        var xSpan = _solidsAxis[^1] - _solidsAxis[0];
        var ySpan = _glycerineAxis[^1] - _glycerineAxis[0];
        var fx = xSpan == 0 ? 0 : (solids - _solidsAxis[0]) / xSpan;
        var fy = ySpan == 0 ? 0 : (glycerine - _glycerineAxis[0]) / ySpan;
        return new PointF(plot.Left + (float)(fx * plot.Width),
                          plot.Bottom - (float)(fy * plot.Height));
    }

    private (int Column, int Row) PointToCell(Rectangle plot, Point point)
    {
        var fx = (point.X - plot.Left) / (double)plot.Width;
        var fy = (plot.Bottom - point.Y) / (double)plot.Height;
        var column = Math.Clamp((int)Math.Round(fx * (_solidsAxis.Length - 1)),
                                0, _solidsAxis.Length - 1);
        var row = Math.Clamp((int)Math.Round(fy * (_glycerineAxis.Length - 1)),
                             0, _glycerineAxis.Length - 1);
        return (column, row);
    }

    protected override void OnMouseMove(MouseEventArgs e)
    {
        base.OnMouseMove(e);
        if (!HasData)
        {
            return;
        }

        _cursor = e.Location;
        _hovering = true;
        Cursor = PlotArea.Contains(e.Location) ? Cursors.Cross : Cursors.Default;
        Invalidate();
    }

    protected override void OnMouseLeave(EventArgs e)
    {
        base.OnMouseLeave(e);
        _hovering = false;
        Invalidate();
    }

    protected override void OnMouseClick(MouseEventArgs e)
    {
        base.OnMouseClick(e);
        if (!HasData || !PlotArea.Contains(e.Location))
        {
            return;
        }

        var (column, row) = PointToCell(PlotArea, e.Location);
        PointPicked?.Invoke(this,
            new RecipePickedEventArgs(_solidsAxis[column], _glycerineAxis[row]));
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _surface?.Dispose();
        }

        base.Dispose(disposing);
    }
}
