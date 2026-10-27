/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        cine: {
          bg: "#08090D",
          secondary: "#101116",
          surface: "#181A21",
          hover: "#232631",
          border: "#292B33",
          red: "#E50914",
          "red-hover": "#B80710",
          text: "#FFFFFF",
          muted: "#A1A1AA",
          success: "#22C55E",
        }
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "sans-serif"],
      },
      boxShadow: {
        'cinema': '0 20px 40px -15px rgba(0, 0, 0, 0.7)',
        'glow-red': '0 0 25px -5px rgba(229, 9, 20, 0.5)',
      }
    },
  },
  plugins: [],
}
