import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./src/app/**/*.{js,ts,jsx,tsx}",
    "./src/components/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["'Space Grotesk'", "Inter", "system-ui"],
      },
      colors: {
        vibe: {
          50: "#e9f2ff",
          100: "#c6dcff",
          200: "#9cc3ff",
          300: "#63a1ff",
          400: "#2f7dff",
          500: "#0d5fd9",
          600: "#0a4cb1",
          700: "#093f8f",
          800: "#0a3875",
          900: "#0b305f",
        },
        ink: "#0b1225",
      },
      boxShadow: {
        floating: "0 20px 60px rgba(15, 76, 153, 0.25)",
      },
      backgroundImage: {
        mesh: "radial-gradient(circle at 20% 20%, rgba(63, 131, 248,0.18), transparent 25%), radial-gradient(circle at 80% 0%, rgba(14, 116, 144,0.18), transparent 18%), radial-gradient(circle at 50% 100%, rgba(59, 130, 246,0.18), transparent 20%)",
      },
    },
  },
  plugins: [],
};

export default config;
