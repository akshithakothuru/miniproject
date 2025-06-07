
export const setupScrollAnimation = () => {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('active');
        }
      });
    },
    { threshold: 0.1 }
  );

  // Observe all elements with the 'reveal' class
  document.querySelectorAll('.reveal').forEach((element) => {
    observer.observe(element);
  });

  // Cleanup function
  return () => {
    document.querySelectorAll('.reveal').forEach((element) => {
      observer.unobserve(element);
    });
  };
};
