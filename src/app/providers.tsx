"use client";
import { PropsWithChildren, useEffect, useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useThemeStore } from "@/stores/theme";
import { useAuthStore } from "@/stores/auth";
import { setActiveUser } from "@/services/api";
import { useLangStore } from "@/stores/lang";
import { setDocumentLang } from "@/utils/i18n";

// Настраиваем общий провайдер для React Query и темы
export const Providers = ({ children }: PropsWithChildren) => {
  const [client] = useState(() => new QueryClient());
  const user = useAuthStore((s) => s.user);
  const lang = useLangStore((s) => s.lang);
  const setLang = useLangStore((s) => s.setLang);

  useEffect(() => {
    // Синхронизируем активного пользователя в API-слое после перезагрузки страницы
    setActiveUser(user);
    if (user?.lang) setLang(user.lang);
  }, [user, setLang]);

  useEffect(() => {
    setDocumentLang(lang);
  }, [lang]);

  return (
    <QueryClientProvider client={client}>
      <ThemeApplier />
      {children}
    </QueryClientProvider>
  );
};

const ThemeApplier = () => {
  const theme = useThemeStore((s) => s.theme);
  // Развешиваем класс на html, чтобы Tailwind darkMode работал
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "dark") root.classList.add("dark");
    else root.classList.remove("dark");
    root.style.setProperty("color-scheme", theme);
  }, [theme]);
  return null;
};
