import type { Metadata, Viewport } from "next";
import { IBM_Plex_Sans, JetBrains_Mono, Noto_Sans_JP } from "next/font/google";
import { LangProvider } from "@/lib/i18n";
import "./globals.css";

const sans = IBM_Plex_Sans({ subsets: ["latin"], weight: ["400", "500", "600"], variable: "--font-sans", display: "swap" });
const mono = JetBrains_Mono({ subsets: ["latin"], weight: ["400", "500", "600"], variable: "--font-mono", display: "swap" });
// Japanese glyphs. The Latin faces above have none, so the browser falls through to this per glyph;
// Google serves the CJK ranges as unicode-range slices, fetched only when a page uses them.
const jp = Noto_Sans_JP({ subsets: ["latin"], weight: ["400", "500"], variable: "--font-jp", display: "swap", preload: false });

export const metadata: Metadata = {
  title: "The Missing Band",
  description:
    "12 kHz to 48 kHz with one 88k-parameter network. Everything above 6 kHz is generated. Pick a speaker, watch the model run in the browser, hear the band come back.",
};

export const viewport: Viewport = { themeColor: "#0A0C10", viewportFit: "cover" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable} ${jp.variable}`}>
      <body><LangProvider>{children}</LangProvider></body>
    </html>
  );
}
