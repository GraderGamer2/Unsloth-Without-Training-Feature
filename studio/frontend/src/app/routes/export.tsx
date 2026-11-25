// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { PERSONAL_VULKAN } from "@/config/personal-build";
import { createRoute, lazyRouteComponent, redirect } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const ExportPage = PERSONAL_VULKAN ? () => null : lazyRouteComponent(
  () => import("@/features/export/export-page"),
  "ExportPage",
);

export type ExportSearch = {
  // Run's output-dir basename, which equals the checkpoint scan's model name.
  run?: string;
  // Preselect an image LoRA (its loras/diffusion file name) in the image LoRA card, from Images > Train.
  imageLora?: string;
};

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/export",
  staticData: { titleKey: "shell.navigation.export" },
  beforeLoad: () => {
    if (PERSONAL_VULKAN) throw redirect({ to: "/chat" });
    return requireAuth();
  },
  validateSearch: (search: Record<string, unknown>): ExportSearch => ({
    run: typeof search.run === "string" ? search.run : undefined,
    imageLora:
      typeof search.imageLora === "string" ? search.imageLora : undefined,
  }),
  component: ExportPage,
});
