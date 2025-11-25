import { NextResponse } from "next/server";
import { getReport } from "@/server/db";

export async function GET(_req: Request, { params }: { params: { id: string } }) {
  const report = await getReport(params.id);
  if (!report) return NextResponse.json({ error: "not_found" }, { status: 404 });
  return NextResponse.json({ report });
}
