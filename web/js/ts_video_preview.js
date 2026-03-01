import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

/**
 * TS Video Preview extension (always-audio)
 * - VHS-like preview (supports /vhs/viewvideo advanced previews when available)
 * - Audio ON by default (no mute feature, no hover-mute)
 * - Robust refresh when tab becomes visible again (fixes stale preview when generated in background)
 */

function chainCallback(object, property, callback) {
  if (!object) return;
  if (property in object && object[property]) {
    const orig = object[property];
    object[property] = function () {
      const r = orig.apply(this, arguments);
      return callback.apply(this, arguments) ?? r;
    };
  } else {
    object[property] = callback;
  }
}

function getSetting(id, fallback) {
  try {
    return app?.ui?.settings?.getSettingValue?.(id) ?? fallback;
  } catch {
    return fallback;
  }
}

function fitHeight(node) {
  try {
    node.setSize?.([node.size[0], node.computeSize?.([node.size[0], node.size[1]])?.[1] ?? node.size[1]]);
  } catch {}
  node?.graph?.setDirtyCanvas?.(true);
  app.graph?.setDirtyCanvas?.(true, true);
}

function startDraggingItems(node, pointer) {
  app.canvas.emitBeforeChange?.();
  app.canvas.graph?.beforeChange?.();
  pointer.finally = () => {
    app.canvas.isDragging = false;
    app.canvas.graph?.afterChange?.();
    app.canvas.emitAfterChange?.();
  };
  app.canvas.processSelect?.(node, pointer.eDown, true);
  app.canvas.isDragging = true;
}

function processDraggedItems(e) {
  if (e.shiftKey || window.LiteGraph?.alwaysSnapToGrid) {
    app.canvas?.graph?.snapToGrid?.(app.canvas.selectedItems);
  }
  app.canvas.dirty_canvas = true;
  app.canvas.dirty_bgcanvas = true;
  app.canvas.onNodeMoved?.(app.canvas.selectedItems?.[0]);
}

function allowDragFromWidget(widget) {
  widget.onPointerDown = function (pointer, node) {
    pointer.onDragStart = () => startDraggingItems(node, pointer);
    pointer.onDragEnd = processDraggedItems;
    app.canvas.dirty_canvas = true;
    return true;
  };
}

function wirePreviewEventsToCanvas(el) {
  const forward = (name, cbName) => {
    el.addEventListener(
      name,
      (e) => {
        e.preventDefault();
        return app.canvas?.[cbName]?.(e);
      },
      true
    );
  };
  forward("contextmenu", "_mousedown_callback");
  forward("pointerdown", "_mousedown_callback");
  forward("mousewheel", "_mousewheel_callback");
  forward("pointermove", "_mousemove_callback");
  forward("pointerup", "_mouseup_callback");

  el.addEventListener("dragover", (e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
    app.dragOverNode = el;
  });
}

function isTSNode(nodeData) {
  return nodeData?.name === "TSVideoCombine" || nodeData?.name === "TSVideoCombineNoMetadata";
}

function getVideoPreviewWidget(node) {
  return node?.widgets?.find?.((x) => x?.name === "videopreview") ?? null;
}

function safeUpdateAllTSPreviews() {
  const nodes = app.graph?._nodes || [];
  for (const n of nodes) {
    const w = getVideoPreviewWidget(n);
    if (!w) continue;

    if (n.__pendingPreviewParams) {
      n.updateParameters?.(n.__pendingPreviewParams, true);
      n.__pendingPreviewParams = null;
      continue;
    }

    try {
      w.updateSource?.();
    } catch {}
  }
  app.graph?.setDirtyCanvas?.(true, true);
}

function installVisibilityHooksOnce() {
  if (window.__tsPreviewVisibilityHookInstalled) return;
  window.__tsPreviewVisibilityHookInstalled = true;

  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) safeUpdateAllTSPreviews();
  });

  window.addEventListener("focus", () => {
    if (!document.hidden) safeUpdateAllTSPreviews();
  });
}

function addVideoPreview(nodeType, isInput = true) {
  chainCallback(nodeType.prototype, "onNodeCreated", function () {
    const node = this;

    const element = document.createElement("div");
    const w = node.addDOMWidget("videopreview", "preview", element, {
      serialize: false,
      hideOnZoom: false,
      getValue() {
        return element.value;
      },
      setValue(v) {
        element.value = v;
      },
    });

    allowDragFromWidget(w);
    wirePreviewEventsToCanvas(element);

    w.value = {
      hidden: false,
      paused: false,
      params: {},
    };

    w.parentEl = document.createElement("div");
    w.parentEl.className = "vhs_preview";
    w.parentEl.style.width = "100%";
    element.appendChild(w.parentEl);

    w.videoEl = document.createElement("video");
    w.videoEl.controls = true;   
    w.videoEl.loop = true;
    w.videoEl.muted = true;  
    w.videoEl.volume = 1.0;
    w.videoEl.style.width = "100%";
    w.videoEl.playsInline = true;

    w.imgEl = document.createElement("img");
    w.imgEl.style.width = "100%";
    w.imgEl.hidden = true;

    w.videoEl.addEventListener("loadedmetadata", () => {
      w.aspectRatio = w.videoEl.videoWidth / w.videoEl.videoHeight;
      fitHeight(node);
    });

    w.imgEl.onload = () => {
      w.aspectRatio = w.imgEl.naturalWidth / w.imgEl.naturalHeight;
      fitHeight(node);
    };

    w.videoEl.addEventListener("error", () => {
      w.parentEl.hidden = true;
      fitHeight(node);
    });

    w.parentEl.appendChild(w.videoEl);
    w.parentEl.appendChild(w.imgEl);

    w.computeSize = function (width) {
      if (this.aspectRatio && !this.parentEl.hidden) {
        const h = (node.size[0] - 20) / this.aspectRatio + 10;
        this.computedHeight = h + 10;
        return [width, h];
      }
      return [width, -4];
    };

    let timeout = null;

    node.updateParameters = (params, forceUpdate) => {
      if (typeof w.value !== "object") w.value = { hidden: false, paused: false, params: {} };
      if (!w.value.params) w.value.params = {};

      const changed = Object.entries(params).some(([k, v]) => w.value.params[k] !== v);
      if (!changed && !forceUpdate) return;

      Object.assign(w.value.params, params);

      if (timeout) clearTimeout(timeout);
      if (forceUpdate) w.updateSource();
      else timeout = setTimeout(() => w.updateSource(), 120);
    };

    w.updateSource = function () {
      if (!this.value?.params) return;

      const params = { ...this.value.params, timestamp: Date.now() };
      this.parentEl.hidden = !!this.value.hidden;

      const fmt = params.format || "";
      const major = fmt.split("/")[0];

      let advp = getSetting("VHS.AdvancedPreviews", "Input Only");
      if (advp === "Never") advp = false;
      else if (advp === "Input Only") advp = !!isInput;
      else advp = true;

      if (major === "video" || fmt === "folder" || (advp && fmt.split("/")[1] === "gif")) {
        this.videoEl.autoplay = !this.value.paused && !this.value.hidden;

        this.videoEl.muted = true;

        if (!advp) {
          this.videoEl.src = api.apiURL("/view?" + new URLSearchParams(params));
        } else {
          let targetWidth = (node.size[0] - 20) * 2 || 256;
          const minW = getSetting("VHS.AdvancedPreviewsMinWidth", 0);
          if (targetWidth < minW) targetWidth = minW;

          if (!params.custom_width || !params.custom_height) {
            params.force_size = targetWidth + "x?";
          } else {
            const ar = params.custom_width / params.custom_height;
            params.force_size = targetWidth + "x" + targetWidth / ar;
          }

          params.deadline = getSetting("VHS.AdvancedPreviewsDeadline", 0);

          this.videoEl.src = api.apiURL("/vhs/viewvideo?" + new URLSearchParams(params));
        }

        this.videoEl.hidden = false;
        this.imgEl.hidden = true;
        return;
      }

      if (major === "image") {
        this.imgEl.src = api.apiURL("/view?" + new URLSearchParams(params));
        this.videoEl.hidden = true;
        this.imgEl.hidden = false;
      }
    };

    w.callback = w.updateSource;
  });
}

function addPreviewOptions(nodeType) {
  chainCallback(nodeType.prototype, "getExtraMenuOptions", function (_, options) {
    const w = getVideoPreviewWidget(this);
    if (!w) return;

    let url = null;

    if (w.videoEl?.hidden === false && w.videoEl?.src) {
      url = api.apiURL("/view?" + new URLSearchParams(w.value.params));
      url = url.replace("%2503d", "001");
    } else if (w.imgEl?.hidden === false && w.imgEl?.src) {
      url = w.imgEl.src;
    }

    const optNew = [];

    if (url) {
      optNew.push(
        {
          content: "Open preview",
          callback: () => window.open(url, "_blank"),
        },
        {
          content: "Save preview",
          callback: () => {
            const a = document.createElement("a");
            a.href = url;
            a.setAttribute("download", w.value.params.filename || "preview");
            document.body.append(a);
            a.click();
            requestAnimationFrame(() => a.remove());
          },
        }
      );
    }

    if (w.videoEl?.hidden === false) {
      optNew.push({
        content: (w.value.paused ? "Resume" : "Pause") + " preview",
        callback: () => {
          if (w.value.paused) w.videoEl?.play();
          else w.videoEl?.pause();
          w.value.paused = !w.value.paused;
        },
      });
    }

    optNew.push({
      content: (w.value.hidden ? "Show" : "Hide") + " preview",
      callback: () => {
        if (!w.videoEl.hidden && !w.value.hidden) w.videoEl.pause();
        else if (w.value.hidden && !w.videoEl.hidden && !w.value.paused) w.videoEl.play();

        w.value.hidden = !w.value.hidden;
        w.parentEl.hidden = w.value.hidden;
        fitHeight(this);
      },
    });

    optNew.push({
      content: "Sync preview",
      callback: () => {
        for (let p of document.getElementsByClassName("vhs_preview")) {
          for (let child of p.children) {
            if (child.tagName === "VIDEO") child.currentTime = 0;
            else if (child.tagName === "IMG") child.src = child.src;
          }
        }
      },
    });


    options.unshift(...optNew);
  });
}

app.registerExtension({
  name: "teskors.utils.ts_video_preview",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!isTSNode(nodeData)) return;

    installVisibilityHooksOnce();

    chainCallback(nodeType.prototype, "onExecuted", function (message) {
      if (message?.gifs?.length) {
        const p = message.gifs[0];

        if (document.hidden) {
          this.__pendingPreviewParams = p;
          return;
        }

        this.updateParameters?.(p, true);
      }
    });

    addVideoPreview(nodeType, false);
    addPreviewOptions(nodeType);
  },
});
