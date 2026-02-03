/**
 * ML Tutorials - Main JavaScript
 * Theme toggle, mobile navigation, and interactivity
 */

(function() {
    'use strict';

    // ========================================
    // DOM Ready
    // ========================================
    document.addEventListener('DOMContentLoaded', function() {
        initThemeToggle();
        initMobileNav();
        initScrollEffects();
        initKaTeX();
        initExercises();
        initCodeBlocks();
        initTableOfContents();
        initSmoothScroll();
    });

    // ========================================
    // Theme Toggle
    // ========================================
    function initThemeToggle() {
        const themeToggle = document.querySelector('.theme-toggle');
        if (!themeToggle) return;

        // Get saved theme or default to system preference
        const savedTheme = localStorage.getItem('theme');
        const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        const initialTheme = savedTheme || (systemPrefersDark ? 'dark' : 'light');
        
        // Apply initial theme
        document.documentElement.setAttribute('data-theme', initialTheme);
        updateThemeToggleIcon(initialTheme);

        // Toggle on click
        themeToggle.addEventListener('click', function() {
            const currentTheme = document.documentElement.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            updateThemeToggleIcon(newTheme);
        });

        // Listen for system theme changes
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) {
            if (!localStorage.getItem('theme')) {
                const newTheme = e.matches ? 'dark' : 'light';
                document.documentElement.setAttribute('data-theme', newTheme);
                updateThemeToggleIcon(newTheme);
            }
        });
    }

    function updateThemeToggleIcon(theme) {
        const sunIcon = document.querySelector('.theme-toggle .sun-icon');
        const moonIcon = document.querySelector('.theme-toggle .moon-icon');
        
        if (sunIcon && moonIcon) {
            if (theme === 'dark') {
                sunIcon.style.display = 'block';
                moonIcon.style.display = 'none';
            } else {
                sunIcon.style.display = 'none';
                moonIcon.style.display = 'block';
            }
        }
    }

    // ========================================
    // Mobile Navigation
    // ========================================
    function initMobileNav() {
        const menuToggle = document.querySelector('.mobile-menu-toggle');
        const nav = document.querySelector('.nav-main');
        const body = document.body;

        if (!menuToggle || !nav) return;

        menuToggle.addEventListener('click', function() {
            const isOpen = nav.classList.toggle('mobile-open');
            menuToggle.classList.toggle('active');
            body.classList.toggle('nav-open', isOpen);
            
            // Accessibility
            menuToggle.setAttribute('aria-expanded', isOpen);
            nav.setAttribute('aria-hidden', !isOpen);
        });

        // Close on escape
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape' && nav.classList.contains('mobile-open')) {
                nav.classList.remove('mobile-open');
                menuToggle.classList.remove('active');
                body.classList.remove('nav-open');
                menuToggle.setAttribute('aria-expanded', 'false');
                nav.setAttribute('aria-hidden', 'true');
            }
        });

        // Close when clicking outside
        document.addEventListener('click', function(e) {
            if (nav.classList.contains('mobile-open') && 
                !nav.contains(e.target) && 
                !menuToggle.contains(e.target)) {
                nav.classList.remove('mobile-open');
                menuToggle.classList.remove('active');
                body.classList.remove('nav-open');
            }
        });

        // Handle dropdown clicks on mobile
        const dropdownTriggers = document.querySelectorAll('.nav-dropdown > a');
        dropdownTriggers.forEach(function(trigger) {
            trigger.addEventListener('click', function(e) {
                if (window.innerWidth <= 768) {
                    e.preventDefault();
                    const parent = this.parentElement;
                    parent.classList.toggle('open');
                }
            });
        });
    }

    // ========================================
    // Scroll Effects
    // ========================================
    function initScrollEffects() {
        const header = document.querySelector('.header');
        if (!header) return;

        let lastScroll = 0;
        let ticking = false;

        window.addEventListener('scroll', function() {
            if (!ticking) {
                window.requestAnimationFrame(function() {
                    handleScroll(header, lastScroll);
                    lastScroll = window.pageYOffset;
                    ticking = false;
                });
                ticking = true;
            }
        });
    }

    function handleScroll(header, lastScroll) {
        const currentScroll = window.pageYOffset;
        
        // Add shadow when scrolled
        if (currentScroll > 50) {
            header.classList.add('scrolled');
        } else {
            header.classList.remove('scrolled');
        }

        // Hide/show header on scroll direction (optional)
        // Uncomment if you want this behavior
        /*
        if (currentScroll > lastScroll && currentScroll > 200) {
            header.classList.add('header-hidden');
        } else {
            header.classList.remove('header-hidden');
        }
        */
    }

    // ========================================
    // KaTeX Auto-render
    // ========================================
    function initKaTeX() {
        if (typeof renderMathInElement === 'function') {
            renderMathInElement(document.body, {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '$', right: '$', display: false},
                    {left: '\\[', right: '\\]', display: true},
                    {left: '\\(', right: '\\)', display: false}
                ],
                throwOnError: false,
                errorColor: '#cc0000',
                strict: false,
                trust: true,
                macros: {
                    "\\R": "\\mathbb{R}",
                    "\\N": "\\mathbb{N}",
                    "\\E": "\\mathbb{E}",
                    "\\P": "\\mathbb{P}",
                    "\\argmax": "\\operatorname{argmax}",
                    "\\argmin": "\\operatorname{argmin}",
                    "\\KL": "\\text{KL}",
                    "\\Var": "\\text{Var}",
                    "\\Cov": "\\text{Cov}"
                }
            });
        }
    }

    // ========================================
    // Exercise Interactivity
    // ========================================
    function initExercises() {
        // Toggle exercise expansion
        const exerciseHeaders = document.querySelectorAll('.exercise-header');
        exerciseHeaders.forEach(function(header) {
            header.addEventListener('click', function() {
                const item = this.closest('.exercise-item');
                item.classList.toggle('open');
            });
        });

        // Toggle solution visibility
        const solutionToggles = document.querySelectorAll('.solution-toggle');
        solutionToggles.forEach(function(toggle) {
            toggle.addEventListener('click', function() {
                const solution = this.nextElementSibling;
                if (solution && solution.classList.contains('solution-content')) {
                    solution.classList.toggle('show');
                    this.textContent = solution.classList.contains('show') 
                        ? 'Hide Solution' 
                        : 'Show Solution';
                }
            });
        });
    }

    // ========================================
    // Code Block Copy
    // ========================================
    function initCodeBlocks() {
        const copyButtons = document.querySelectorAll('.code-block-copy');
        copyButtons.forEach(function(button) {
            button.addEventListener('click', function() {
                const codeBlock = this.closest('.code-block');
                const code = codeBlock.querySelector('code');
                
                if (code) {
                    navigator.clipboard.writeText(code.textContent).then(function() {
                        button.textContent = 'Copied!';
                        setTimeout(function() {
                            button.textContent = 'Copy';
                        }, 2000);
                    }).catch(function() {
                        button.textContent = 'Failed';
                        setTimeout(function() {
                            button.textContent = 'Copy';
                        }, 2000);
                    });
                }
            });
        });
    }

    // ========================================
    // Table of Contents
    // ========================================
    function initTableOfContents() {
        const toc = document.querySelector('.toc-container');
        const article = document.querySelector('.article-content');
        
        if (!toc || !article) return;

        // Generate TOC from headings
        const headings = article.querySelectorAll('h2, h3');
        const tocList = toc.querySelector('.toc-list');
        
        if (tocList && headings.length > 0) {
            headings.forEach(function(heading, index) {
                // Add ID if not present
                if (!heading.id) {
                    heading.id = 'section-' + index;
                }

                const link = document.createElement('a');
                link.href = '#' + heading.id;
                link.className = 'toc-link toc-' + heading.tagName.toLowerCase();
                link.textContent = heading.textContent;
                tocList.appendChild(link);
            });

            // Show TOC when there are enough headings
            if (headings.length >= 3) {
                toc.classList.add('visible');
            }

            // Highlight current section on scroll
            const tocLinks = toc.querySelectorAll('.toc-link');
            
            const observerOptions = {
                rootMargin: '-20% 0px -80% 0px'
            };

            const observer = new IntersectionObserver(function(entries) {
                entries.forEach(function(entry) {
                    if (entry.isIntersecting) {
                        tocLinks.forEach(function(link) {
                            link.classList.remove('active');
                            if (link.getAttribute('href') === '#' + entry.target.id) {
                                link.classList.add('active');
                            }
                        });
                    }
                });
            }, observerOptions);

            headings.forEach(function(heading) {
                observer.observe(heading);
            });
        }
    }

    // ========================================
    // Smooth Scroll
    // ========================================
    function initSmoothScroll() {
        document.querySelectorAll('a[href^="#"]').forEach(function(anchor) {
            anchor.addEventListener('click', function(e) {
                const targetId = this.getAttribute('href');
                if (targetId === '#') return;
                
                const target = document.querySelector(targetId);
                if (target) {
                    e.preventDefault();
                    const headerHeight = document.querySelector('.header')?.offsetHeight || 0;
                    const targetPosition = target.getBoundingClientRect().top + window.pageYOffset;
                    
                    window.scrollTo({
                        top: targetPosition - headerHeight - 20,
                        behavior: 'smooth'
                    });
                }
            });
        });
    }

    // ========================================
    // Utility Functions
    // ========================================
    window.MLTutorials = {
        // Toggle dark/light theme programmatically
        setTheme: function(theme) {
            document.documentElement.setAttribute('data-theme', theme);
            localStorage.setItem('theme', theme);
            updateThemeToggleIcon(theme);
        },

        // Get current theme
        getTheme: function() {
            return document.documentElement.getAttribute('data-theme');
        },

        // Re-render math (useful after dynamic content load)
        renderMath: function(element) {
            if (typeof renderMathInElement === 'function') {
                renderMathInElement(element || document.body, {
                    delimiters: [
                        {left: '$$', right: '$$', display: true},
                        {left: '$', right: '$', display: false}
                    ],
                    throwOnError: false
                });
            }
        }
    };

})();
