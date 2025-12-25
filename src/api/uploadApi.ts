// src/api/uploadApi.ts
import { http } from "./httpClient";

export function uploadDocument(
  file: File,
  namespace: string,
  userId: string
) {
  const formData = new FormData();
  formData.append("file", file);

  return http.post(
    `/v1/documents/upload`,
    formData,
    {
      params: { namespace, user_id: userId },
      headers: { "Content-Type": "multipart/form-data" },
    }
  );
}
