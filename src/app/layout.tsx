import type { Metadata } from "next";
import type { ReactNode } from "react";
import "@/styles/globals.css";
import { Providers } from "./providers";
import { HeaderBar } from "@/components/Layout/HeaderBar";

export const metadata: Metadata = {
  title: "Vibe Interview Platform",
  description: "Готовый фронтенд для тех. собеседований с гибким API-слоем",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="ru">
      <body className="min-h-screen bg-mesh bg-cover bg-fixed">
        <Providers>
          <div className="mx-auto w-full max-w-[1600px] px-1 sm:px-2 py-4">
            <HeaderBar />
            {children}
          </div>
        </Providers>
      </body>
    </html>
  );
}
