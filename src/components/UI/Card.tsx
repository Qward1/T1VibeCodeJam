import { ReactNode } from "react";
import { cn } from "@/utils";

type Props = {
  title?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
} & React.HTMLAttributes<HTMLDivElement>;

export const Card = ({ title, actions, children, className, ...props }: Props) => (
  <div
    className={cn(
      "glass rounded-2xl border border-[var(--border)] bg-[var(--card)] p-5 shadow-lg shadow-vibe-900/5",
      className
    )}
    {...props}
  >
    {(title || actions) && (
      <div className="mb-4 flex items-center justify-between">
        {title && <h3 className="text-lg font-semibold">{title}</h3>}
        {actions}
      </div>
    )}
    {children}
  </div>
);
