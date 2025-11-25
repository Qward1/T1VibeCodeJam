"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { useMutation } from "@tanstack/react-query";
import { api } from "@/services/api";
import { Input } from "@/components/UI/Input";
import { Button } from "@/components/UI/Button";
import Link from "next/link";
import { useTranslation } from "@/utils/i18n";

export default function RegisterPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState<string | null>(null);
  const { t, lang } = useTranslation();

  const mutation = useMutation({
    mutationFn: () => api.register({ email, password, name, lang }),
    onSuccess: () => {
      // После успешной регистрации отправляем на страницу входа
      router.push("/auth/login");
    },
    onError: (err: any) => {
      console.error(err);
      setError(err?.message || "Не удалось создать аккаунт. Проверьте соединение или попробуйте другой email.");
    },
  });

  const isValid = password.length >= 6 && password === confirm && email.includes("@");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    if (isValid) mutation.mutate();
  };

  return (
    <div className="mx-auto flex min-h-[70vh] max-w-md flex-col justify-center rounded-3xl border border-[var(--border)] bg-[var(--card)] p-8 shadow-floating">
      <div className="mb-6 text-center">
        <div className="inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-vibe-500 to-vibe-700 text-2xl text-white shadow-floating">
          ✨
        </div>
        <h1 className="mt-4 text-2xl font-semibold">{t("registerTitle")}</h1>
        <p className="text-sm text-[var(--muted)]">{t("registerDesc")}</p>
      </div>
      <form className="space-y-4" onSubmit={handleSubmit}>
        {error && (
          <div className="rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-700 dark:border-amber-900/40 dark:bg-amber-900/10 dark:text-amber-200">
            {error}
          </div>
        )}
        <Input label="Имя" value={name} onChange={(e) => setName(e.target.value)} required />
        <Input label="Email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
        <Input label="Пароль" type="password" value={password} onChange={(e) => setPassword(e.target.value)} required />
        <Input
          label="Повторите пароль"
          type="password"
          value={confirm}
          onChange={(e) => setConfirm(e.target.value)}
          required
          hint={password && confirm && password !== confirm ? "Пароли не совпадают" : undefined}
        />
        <Button type="submit" disabled={!isValid || mutation.isPending} className="w-full">
          {mutation.isPending ? "Регистрируем..." : t("registerButton")}
        </Button>
        {!isValid && password && confirm && (
          <p className="text-sm text-amber-600 dark:text-amber-300">Пароли должны совпадать и быть не короче 6 символов.</p>
        )}
        {error && <p className="text-sm text-amber-600 dark:text-amber-300">{error}</p>}
      </form>
      <div className="mt-4 text-center text-sm text-[var(--muted)]">
        Уже есть аккаунт? <Link href="/auth/login" className="text-vibe-600">{t("navLogin")}</Link>
      </div>
    </div>
  );
}
