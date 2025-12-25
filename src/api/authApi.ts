import { http } from "./httpClient";

export interface LoginRequest {
  email: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  expires_in: number;
}

export async function login(
  payload: LoginRequest
): Promise<LoginResponse> {
  const response = await http.post<LoginResponse>(
    "/v1/auth/login",
    payload
  );
  return response.data;
}

export async function logout(): Promise<void> {
  await http.post("/v1/auth/logout");
}

export async function getCurrentUser() {
  const response = await http.get("/v1/auth/me");
  return response.data;
}
