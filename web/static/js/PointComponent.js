class PointComponent {
  constructor(index, container) {
    this.index = index;
    this.container = container;
    this.init();
  }

  init() {
    this.el = document.createElement('div');
    this.el.className = 'point-component';
    this.el.dataset.index = this.index;
    this.el.textContent = `Point ${this.index}`;
    this.bindEvents();
    this.container.appendChild(this.el);
  }

  bindEvents() {
    this.el.addEventListener('click', (e) => {
      this.onClick();
    });
  }

  onClick() {
    // Trigger point click event in communication protocol
    console.log(`Point ${this.index} clicked`);
    // Placeholder: dispatch event
    const event = new CustomEvent('pointSelected', { detail: { index: this.index } });
    window.dispatchEvent(event);
  }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = PointComponent;
}

window.PointComponent = PointComponent;
