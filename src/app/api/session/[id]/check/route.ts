import { NextResponse } from "next/server";

export async function POST() {
  // Простейший эмулятор проверки — всегда один падающий тест для демонстрации
  const result = {
    passed: false,
    summary: "Тест №3 падает на кейсе с пустым массивом",
    cases: [
      { name: "Возвращает индексы", passed: true },
      { name: "Работает с дубликатами", passed: true },
      { name: "Пустой массив", passed: false, details: "ожидалось []" },
    ],
  };
  return NextResponse.json({ result });
}
