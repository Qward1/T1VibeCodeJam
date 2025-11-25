"use client";
import { ButtonHTMLAttributes, ReactNode } from "react";
import { cn } from "@/utils";

type Variant = "primary" | "ghost" | "outline";
type Size = "sm" | "md" | "lg";

type Props = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant;
  size?: Size;
  iconLeft?: ReactNode;
  iconRight?: ReactNode;
};

export const Button = ({
  children,
  className,
  variant = "primary",
  size = "md",
  iconLeft,
  iconRight,
  ...props
}: Props) => {
  const base = "inline-flex items-center gap-2 rounded-full text-sm font-semibold transition";
  const sizes: Record<Size, string> = {
    sm: "px-4 py-2",
    md: "px-5 py-2.5",
    lg: "px-6 py-3",
  };
  const variants: Record<Variant, string> = {
    primary:
      "bg-gradient-to-r from-vibe-500 to-vibe-700 text-white shadow-floating hover:translate-y-[-1px]",
    ghost: "bg-transparent text-vibe-500 hover:bg-vibe-50 dark:hover:bg-white/10",
    outline:
      "border border-vibe-300 text-vibe-600 hover:bg-vibe-50 dark:border-white/20 dark:text-white/90 dark:hover:bg-white/5",
  };

  return (
    <button className={cn(base, sizes[size], variants[variant], className)} {...props}>
      {iconLeft}
      {children}
      {iconRight}
    </button>
  );
};
