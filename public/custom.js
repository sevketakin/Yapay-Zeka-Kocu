(function () {
  function ekle(tag, ozellikler) {
    var el = document.createElement(tag);
    Object.keys(ozellikler).forEach(function (k) {
      el.setAttribute(k, ozellikler[k]);
    });
    document.head.appendChild(el);
  }

  ekle("link", { rel: "manifest", href: "/public/manifest.json" });
  ekle("link", { rel: "apple-touch-icon", href: "/public/icon-192.png" });
  ekle("meta", { name: "theme-color", content: "#0f172a" });
  ekle("meta", { name: "apple-mobile-web-app-capable", content: "yes" });
  ekle("meta", { name: "apple-mobile-web-app-status-bar-style", content: "black-translucent" });
  ekle("meta", { name: "apple-mobile-web-app-title", content: "Koçum" });
})();
