// Tailwind does layout only; colours, type and rules follow the paper's
// figure style (scripts/paper_figures/_style.py, mirrored in static/styles.css).
// Built to a static file so the demo has no CDN dependency at run time:
//   bash inference/frontend/build_css.sh   ->  static/tailwind.css
// Dark-theme utility classes still emitted by app.js (slate, rose, amber) are
// mapped onto the paper's INK / GREY / vermilion / orange.
const INK = '#1A1A1A', GREY = '#8C8C8C';
const ramp = (mid, dark) => ({ 50: dark, 100: dark, 200: dark, 300: dark, 400: mid,
  500: mid, 600: dark, 700: dark, 800: dark, 900: dark, 950: dark });
module.exports = {
  content: [__dirname + '/static/index.html', __dirname + '/static/app.js'],
  theme: {
    borderRadius: { none: '0', DEFAULT: '0', sm: '0', md: '0', lg: '0', xl: '0', '2xl': '0', full: '9999px' },
    boxShadow: { none: 'none', DEFAULT: 'none', sm: 'none', md: 'none', lg: 'none', xl: 'none' },
    extend: {
      fontFamily: {
        sans: ['Arial', 'Helvetica', '"Liberation Sans"', '"DejaVu Sans"', 'sans-serif'],
        mono: ['"DejaVu Sans Mono"', 'Menlo', 'Consolas', '"Liberation Mono"', 'monospace'],
      },
      colors: {
        slate: ramp(GREY, INK),
        rose: ramp('#D55E00', '#D55E00'),
        amber: ramp('#E69F00', INK),
      },
    },
  },
};
