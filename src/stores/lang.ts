import { create } from "zustand";
import { persist } from "zustand/middleware";

export type Lang = "ru" | "en";

type LangState = {
  lang: Lang;
  setLang: (lang: Lang) => void;
};

export const useLangStore = create<LangState>()(
  persist(
    (set) => ({
      lang: "ru",
      setLang: (lang) => set({ lang }),
    }),
    { name: "vibe-lang" }
  )
);
