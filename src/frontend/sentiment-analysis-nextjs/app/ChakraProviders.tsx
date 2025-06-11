"use client";
import { ChakraProvider } from "@chakra-ui/react";

export default function ChakraProviders({ children }: { children: React.ReactNode }) {
  return <ChakraProvider>{children}</ChakraProvider>;
}