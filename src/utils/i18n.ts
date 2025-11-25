import { useLangStore, Lang } from "@/stores/lang";

type Dictionary = Record<string, { ru: string; en: string }>;

const translations: Dictionary = {
  navTopics: { ru: "Темы", en: "Topics" },
  navProfile: { ru: "Профиль", en: "Profile" },
  navAdmin: { ru: "Админ", en: "Admin" },
  navLogin: { ru: "Войти", en: "Login" },
  navRegister: { ru: "Зарегистрироваться", en: "Register" },
  heroTitle: {
    ru: "Платформа технических собеседований с акцентом на синий и скорость",
    en: "Tech interview platform focused on blue aesthetics and speed",
  },
  heroDesc: {
    ru: "Полностью компонентный frontend: auth, профиль, выбор тем, лайв-сессия, отчёты и админка. Один API-слой — любая backend-подмена.",
    en: "Fully componentized frontend: auth, profile, topic selection, live interview session, reports and admin. Single API layer to swap any backend.",
  },
  heroStart: { ru: "Начать интервью", en: "Start interview" },
  heroRegister: { ru: "Регистрация", en: "Register" },
  loginTitle: { ru: "Вход", en: "Login" },
  loginDesc: { ru: "Используйте демо-данные или свои.", en: "Use demo credentials or your own." },
  loginButton: { ru: "Войти", en: "Sign in" },
  registerTitle: { ru: "Регистрация", en: "Register" },
  registerDesc: { ru: "Минимум полей — максимум скорости.", en: "Minimal fields, maximum speed." },
  registerButton: { ru: "Создать аккаунт", en: "Create account" },
  changePassword: { ru: "Сменить пароль", en: "Change password" },
  changeLanguage: { ru: "Сменить язык", en: "Change language" },
};

export const useTranslation = () => {
  const lang = useLangStore((s) => s.lang);
  const t = (key: keyof typeof translations) => translations[key]?.[lang] ?? translations[key]?.ru ?? key;
  return { t, lang };
};

export const setDocumentLang = (lang: Lang) => {
  if (typeof document !== "undefined") {
    document.documentElement.lang = lang;
  }
};
