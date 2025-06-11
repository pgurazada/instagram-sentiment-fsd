"use client";

import { useState } from "react";
import {
  Box,
  Button,
  Heading,
  Input,
  Text,
  VStack,
  Container,
} from "@chakra-ui/react";

const SentimentForm = () => {
  const [textInput, setTextInput] = useState("🙏💙💚");
  const [result, setResult] = useState<{ sentiment: string; score: number } | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: textInput }),
      });
      const data = await res.json();
      setResult({
        sentiment: data.data.sentiment_label,
        score: data.data.sentiment_score,
      });
    } catch {
      alert("Failed to fetch sentiment.");
    }
    setLoading(false);
  };

  return (
    <Container centerContent minH="100vh" bg="gray.900" maxW="100vw" py={20}>
      <Box
        bg="white"
        p={8}
        rounded="xl"
        shadow="2xl"
        w="100%"
        maxW="lg"
        textAlign="center"
      >
        <Heading mb={2} color="gray.800">
          Instagram Comments Sentiment Analyzer
        </Heading>
        <Text mb={6} color="gray.500">
          Evaluate the sentiment of comments received on Instagram marketing campaigns.
        </Text>
        <form onSubmit={handleSubmit}>
          <VStack spacing={4}>
            <Input
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              placeholder="Enter a comment"
              size="lg"
              bg="gray.100"
            />
            <Button
              colorScheme="blue"
              type="submit"
              isLoading={loading}
              w="100%"
              size="lg"
              fontWeight="bold"
            >
              Analyze Sentiment
            </Button>
          </VStack>
        </form>
        {result && (
          <Box mt={6} p={4} bg="purple.50" rounded="md">
            <Text fontWeight="bold">
              Sentiment: <span style={{ textTransform: "capitalize" }}>{result.sentiment}</span>
            </Text>
            <Text>Score: {result.score}</Text>
          </Box>
        )}
      </Box>
    </Container>
  );
};

export default SentimentForm;