import { create } from "zustand";
import { persist } from "zustand/middleware";
import { User } from "@/types";

type AuthState = {
  user?: User;
  setUser: (user?: User) => void;
  logout: () => void;
};

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: undefined,
      setUser: (user) => set({ user }),
      logout: () => set({ user: undefined }),
    }),
    { name: "vibe-auth" }
  )
);
