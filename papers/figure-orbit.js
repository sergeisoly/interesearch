// Orthographic views for small diagrams of vectors, with z up in model space.
window.ResearchOrbit = (() => {
  function project([x, y, z], { yaw, pitch }) {
    const right = x * Math.cos(yaw) - y * Math.sin(yaw);
    const depth = x * Math.sin(yaw) + y * Math.cos(yaw);
    return [right, depth * Math.sin(pitch) - z * Math.cos(pitch)];
  }

  function bind(surfaces, draw, initial = { yaw: 0.6, pitch: 0.35 }) {
    let camera = { ...initial };
    let drag = null;
    function redraw() {
      draw(camera);
    }
    function rotate(yaw, pitch) {
      camera = {
        yaw: (camera.yaw + yaw) % (2 * Math.PI),
        pitch: Math.min(1.35, Math.max(-1.35, camera.pitch + pitch)),
      };
      redraw();
    }
    function reset() {
      camera = { ...initial };
      redraw();
    }

    for (const surface of surfaces) {
      surface.addEventListener("pointerdown", event => {
        if (drag || !event.isPrimary || event.button !== 0) return;
        drag = { id: event.pointerId, x: event.clientX, y: event.clientY };
        surface.setPointerCapture(event.pointerId);
      });
      surface.addEventListener("pointermove", event => {
        if (drag?.id !== event.pointerId) return;
        const dx = event.clientX - drag.x;
        const dy = event.clientY - drag.y;
        drag.x = event.clientX;
        drag.y = event.clientY;
        rotate(dx * 0.012, dy * 0.012);
      });
      const endDrag = event => {
        if (drag?.id === event.pointerId) drag = null;
      };
      for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) {
        surface.addEventListener(type, endDrag);
      }
      surface.addEventListener("keydown", event => {
        const step = Math.PI / 18;
        switch (event.key) {
          case "ArrowLeft": rotate(-step, 0); break;
          case "ArrowRight": rotate(step, 0); break;
          case "ArrowUp": rotate(0, step); break;
          case "ArrowDown": rotate(0, -step); break;
          case "Home": reset(); break;
          default: return;
        }
        event.preventDefault();
      });
    }
    return { redraw, reset, rotate };
  }
  return { bind, project };
})();
