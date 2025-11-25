"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { useMutation } from "@tanstack/react-query";
import { api } from "@/services/api";
import { Card } from "@/components/UI/Card";
import { Input } from "@/components/UI/Input";
import { Button } from "@/components/UI/Button";

export default function ChangePasswordPage() {
  const router = useRouter();
  const [oldPass, setOldPass] = useState("");
  const [newPass, setNewPass] = useState("");
  const [confirmPass, setConfirmPass] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const mutation = useMutation({
    mutationFn: () => api.changePassword(oldPass, newPass),
    onSuccess: () => {
      setSuccess(true);
      setError(null);
      setTimeout(() => router.push("/profile"), 800);
    },
    onError: (err: any) => {
      setSuccess(false);
      setError(err?.message ?? "Не удалось сменить пароль");
    },
  });

  const disabled =
    mutation.isPending || !oldPass || newPass.length < 6 || newPass !== confirmPass;

  return (
    <main className="flex min-h-[70vh] items-center justify-center">
      <Card className="w-full max-w-md" title="Сменить пароль">
        <p className="mb-4 text-sm text-[var(--muted)]">
          Введите текущий пароль и новый дважды. После успешной смены вы останетесь в аккаунте.
        </p>
        <div className="space-y-3">
          <Input
            label="Текущий пароль"
            type="password"
            value={oldPass}
            onChange={(e) => setOldPass(e.target.value)}
          />
          <Input
            label="Новый пароль"
            type="password"
            value={newPass}
            onChange={(e) => setNewPass(e.target.value)}
            hint="Минимум 6 символов"
          />
          <Input
            label="Повторите новый пароль"
            type="password"
            value={confirmPass}
            onChange={(e) => setConfirmPass(e.target.value)}
            hint={confirmPass && newPass !== confirmPass ? "Пароли не совпадают" : undefined}
          />
          {error && (
            <div className="rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-700 dark:border-amber-900/40 dark:bg-amber-900/10 dark:text-amber-200">
              {error}
            </div>
          )}
          {success && (
            <div className="rounded-xl border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm text-emerald-700 dark:border-emerald-900/40 dark:bg-emerald-900/10 dark:text-emerald-200">
              Пароль успешно изменён
            </div>
          )}
          <Button className="w-full" disabled={disabled} onClick={() => mutation.mutate()}>
            {mutation.isPending ? "Меняем..." : "Сменить пароль"}
          </Button>
        </div>
      </Card>
    </main>
  );
}
