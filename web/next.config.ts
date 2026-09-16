import type { NextConfig } from "next";
import path from "node:path";

const nextConfig: NextConfig = {
  // the app lives in web/ inside a larger research repo; pin the root so a stray lockfile above us is ignored
  turbopack: { root: path.resolve(__dirname) },
};

export default nextConfig;
