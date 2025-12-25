import { http } from "./httpClient";

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  session_id?: string;
  message: string;
  namespace: string;
}

export interface ChatResponse {
  session_id: string;
  answer: string;
  sources?: Array<{
    document_id: string;
    chunk_id: string;
  }>;
}

export async function sendChatMessage(
  payload: ChatRequest
): Promise<ChatResponse> {
  const response = await http.post<ChatResponse>(
    "/v1/chat",
    payload
  );
  return response.data;
}
