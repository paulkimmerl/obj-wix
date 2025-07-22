const tower = document.querySelector('.tower');
const blocks = document.querySelectorAll('.block');

window.addEventListener('scroll', () => {
  const scrollY = window.scrollY;
  // Move the tower opposite to scroll, and scale depth
  tower.style.transform = `translateZ(${scrollY * -1}px)`;
});

// Optional: Set a tall body so we can scroll
document.body.style.height = `${blocks.length * 100}vh`;
