const tower = document.querySelector('.tower');
const blocks = document.querySelectorAll('.block');

// Positioniere jeden Block in Z-Richtung (wie Turm)
blocks.forEach((block, i) => {
  const depth = i * -400; // Abstand zwischen Etagen
  block.style.transform = `translateZ(${depth}px)`;
});

window.addEventListener('scroll', () => {
  const scrollY = window.scrollY;
  tower.style.transform = `translate(-50%, -50%) translateZ(${scrollY * -2}px)`;
});

// Body-Höhe für Scroll erzeugen
document.body.style.height = `${blocks.length * 400}px`;
