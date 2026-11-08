import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Vani — Document Chat",
  description: "Search your documents and review relevant source passages.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
