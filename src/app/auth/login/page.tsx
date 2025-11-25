"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { useMutation } from "@tanstack/react-query";
import { api } from "@/services/api";
import { Input } from "@/components/UI/Input";
import { Button } from "@/components/UI/Button";
import { useAuthStore } from "@/stores/auth";
import Link from "next/link";
import { useTranslation } from "@/utils/i18n";

export default function LoginPage() {
  const router = useRouter();
  const setUser = useAuthStore((s) => s.setUser);
  const [email, setEmail] = useState("user@vibe.dev");
  const [password, setPassword] = useState("password");
  const [error, setError] = useState<string | null>(null);
  const { t } = useTranslation();

  const mutation = useMutation({
    mutationFn: () => api.login({ email, password }),
    onSuccess: (user) => {
      setError(null);
      setUser(user);
      router.push("/profile");
    },
    onError: () => {
      setError("Неверный логин или пароль");
    },
  });

  return (
    <div className="mx-auto flex min-h-[70vh] max-w-md flex-col justify-center rounded-3xl border border-[var(--border)] bg-[var(--card)] p-8 shadow-floating">
      <div className="mb-6 text-center">
        <div className="inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-vibe-500 to-vibe-700 text-2xl text-white shadow-floating">
          🔐
        </div>
        <h1 className="mt-4 text-2xl font-semibold">{t("loginTitle")}</h1>
        <p className="text-sm text-[var(--muted)]">{t("loginDesc")}</p>
      </div>
      <form className="space-y-4" onSubmit={(e) => e.preventDefault()}>
        {error && (
          <div className="rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-700 dark:border-amber-900/40 dark:bg-amber-900/10 dark:text-amber-200">
            {error}
          </div>
        )}
        <Input
          label="Email"
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
        <Input
          label="Пароль"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        <Button onClick={() => mutation.mutate()} disabled={mutation.isPending} className="w-full">
          {mutation.isPending ? "Входим..." : t("loginButton")}
        </Button>
      </form>
      <div className="mt-4 text-center text-sm text-[var(--muted)]">
        Нет аккаунта? <Link href="/auth/register" className="text-vibe-600">{t("navRegister")}</Link>
      </div>
    </div>
  );
}
