"use client";
import Link from "next/link";
import Image from "next/image";
import { useRouter } from "next/navigation";
import { Button } from "@/components/UI/Button";
import { ThemeToggle } from "@/components/UI/ThemeToggle";
import { useAuthStore } from "@/stores/auth";
import { useLangStore } from "@/stores/lang";
import { useTranslation } from "@/utils/i18n";
import { useSessionStore } from "@/stores/session";

// Клиентская шапка: показывает кнопки в зависимости от состояния сессии
export const HeaderBar = () => {
  const user = useAuthStore((s) => s.user);
  const isAdmin = !!user?.admin || user?.role === "superadmin";
  const { t } = useTranslation();
  const { lang, setLang } = useLangStore();
  const router = useRouter();
  const interviewId = useSessionStore((s) => s.interviewId);
  const setSession = useSessionStore((s) => s.setSession);
  const setInterviewId = useSessionStore((s) => s.setInterviewId);
  const resetSession = useSessionStore((s) => s.reset);

  return (
    <header className="mb-8 flex items-center justify-between rounded-2xl border border-[var(--border)] bg-[var(--card)]/80 px-4 py-3 shadow-lg shadow-vibe-900/5 backdrop-blur">
      <Link href="/" className="flex items-center gap-3 font-semibold text-vibe-700 dark:text-white">
        <span className="flex h-12 w-12 items-center justify-center overflow-hidden rounded-xl border border-[var(--border)] bg-[var(--card)] shadow-floating">
          <Image
            key="logo-new"
            src="/logo-new.png"
            alt="Логотип"
            width={48}
            height={48}
            className="h-full w-full object-cover"
            priority
          />
        </span>
        <div>
          <div>AI INTERVIEW PLATFORM</div>
          <div className="text-xs text-[var(--muted)]">BY T1</div>
        </div>
      </Link>
      <div className="flex flex-wrap items-center gap-3 text-sm">
        <button
          onClick={async () => {
            if (!user) return router.push("/auth/register");
            try {
              const active = await api.getActiveInterview();
              if (active?.id) {
                setSession(active);
                setInterviewId(active.id);
                return router.push(`/interview/session/${active.id}`);
              }
              // Активной нет — очищаем локальный хвост и идём на выбор
              resetSession();
              return router.push("/interview/select");
            } catch (e) {
              resetSession();
              return router.push("/interview/select");
            }
          }}
          className="rounded-full px-3 py-2 text-[var(--muted)] hover:text-vibe-600 dark:hover:text-white"
        >
          Собеседование
        </button>
        {user && (
          <Link href="/profile" className="rounded-full px-3 py-2 text-[var(--muted)] hover:text-vibe-600 dark:hover:text-white">
            {t("navProfile")}
          </Link>
        )}
        {isAdmin && (
          <Link href="/admin" className="rounded-full px-3 py-2 text-[var(--muted)] hover:text-vibe-600 dark:hover:text-white">
            {t("navAdmin")}
          </Link>
        )}
        {!user && (
          <>
            <Link href="/auth/login" className="rounded-full px-3 py-2 text-[var(--muted)] hover:text-vibe-600 dark:hover:text-white">
              {t("navLogin")}
            </Link>
            <Link href="/auth/register">
              <Button variant="outline" className="px-4 py-2">
                {t("navRegister")}
              </Button>
            </Link>
          </>
        )}
        <div className="flex items-center gap-1 rounded-full border border-[var(--border)] px-2 py-1 text-xs">
          <button
            className={`rounded-full px-2 py-1 ${lang === "ru" ? "bg-vibe-500 text-white" : "text-[var(--muted)] hover:text-vibe-600"}`}
            onClick={() => setLang("ru")}
          >
            RU
          </button>
          <button
            className={`rounded-full px-2 py-1 ${lang === "en" ? "bg-vibe-500 text-white" : "text-[var(--muted)] hover:text-vibe-600"}`}
            onClick={() => setLang("en")}
          >
            EN
          </button>
        </div>
        <ThemeToggle />
      </div>
    </header>
  );
};
