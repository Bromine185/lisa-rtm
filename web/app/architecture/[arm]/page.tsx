import { notFound } from "next/navigation";
import { MODEL_ORDER, isModel, isReleased } from "@/lib/arms";
import { ArchPage } from "@/components/ArchPage";

// the eight arms and the four released models; the segment keeps its name
export function generateStaticParams() {
  return MODEL_ORDER.map((arm) => ({ arm }));
}

export default async function Page({ params }: { params: Promise<{ arm: string }> }) {
  const { arm } = await params;
  if (!isModel(arm)) notFound();
  // a released model is a fresh page (its own fetch, stage strip and block scene); the arms share one, so
  // moving between them keeps the WebGL context and the camera and the scene switches in place
  return <ArchPage key={isReleased(arm) ? arm : "lisa"} model={arm} />;
}
