import { notFound } from "next/navigation";
import { ARM_ORDER, isArm } from "@/lib/arms";
import { ArchPage } from "@/components/ArchPage";

export function generateStaticParams() {
  return ARM_ORDER.map((arm) => ({ arm }));
}

export default async function Page({ params }: { params: Promise<{ arm: string }> }) {
  const { arm } = await params;
  if (!isArm(arm)) notFound();
  return <ArchPage arm={arm} />;
}
