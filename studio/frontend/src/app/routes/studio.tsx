// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { PERSONAL_VULKAN } from "@/config/personal-build";
import { createRoute, lazyRouteComponent, redirect } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const StudioPage = PERSONAL_VULKAN ? () => null : lazyRouteComponent(
  () => import("@/features/studio/studio-page"),
  "StudioPage",
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/studio",
  staticData: { titleKey: "studio.routeTitle" },
  beforeLoad: () => {
    if (PERSONAL_VULKAN) throw redirect({ to: "/chat" });
    return requireAuth();
  },
  component: StudioPage,
});
