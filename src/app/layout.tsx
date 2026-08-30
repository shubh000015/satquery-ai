import type { Metadata } from "next";
import { IBM_Plex_Mono, Inter, Oswald } from "next/font/google";
import "./globals.css";

const sans = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

const display = Oswald({
  variable: "--font-oswald",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

const plex = IBM_Plex_Mono({
  variable: "--font-plex",
  subsets: ["latin"],
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "SATQUERY · SIH",
  description:
    "SATQUERY AI — an interactive vision-language assistant for multimodal remote sensing image analysis through text queries. Optical, SAR, bi-temporal. Smart India Hackathon 2026, ISRO.",
  icons: { icon: "/favicon.svg" },
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${sans.variable} ${display.variable} ${plex.variable} scroll-smooth antialiased`}
    >
      <body className="bg-void text-ink">{children}</body>
    </html>
  );
}


