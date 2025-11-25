import { NextResponse } from "next/server";
import { saveChat, getChat } from "@/server/db";
import { Message } from "@/types";

const uid = () => Math.random().toString(36).slice(2, 10);

export async function POST(req: Request, { params }: { params: { id: string } }) {
  const body = await req.json();
  const { message, role = "user" } = body as { message: string; role?: Message["role"] };
  if (!message) return NextResponse.json({ error: "empty" }, { status: 400 });
  const userMsg: Message = { id: uid(), role: role ?? "user", content: message, createdAt: new Date().toISOString() };
  await saveChat(params.id, userMsg);
  const aiMsg: Message = {
    id: uid(),
    role: "assistant",
    content: "Проверьте крайние случаи: пустые входные данные и большие объёмы. Что по сложности?",
    createdAt: new Date().toISOString(),
  };
  await saveChat(params.id, aiMsg);
  return NextResponse.json({ reply: aiMsg });
}

export async function GET(_req: Request, { params }: { params: { id: string } }) {
  const chat = await getChat(params.id);
  return NextResponse.json({ chat });
}
