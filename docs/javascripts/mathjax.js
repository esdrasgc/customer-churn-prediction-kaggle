window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$','$$'], ['\\[','\\]']]
  },
  options: {
    skipHtmlTags: ['script','noscript','style','textarea','pre','code']
  }
};

// Reprocessa matemática após trocas de página (Material navigation.instant)
document$.subscribe(() => {
  MathJax.typesetPromise?.();
});
