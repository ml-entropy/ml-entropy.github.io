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
        initMobileSidebar();
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
        const menuToggle = document.getElementById('navToggle');
        const nav = document.getElementById('navMenu');
        const body = document.body;

        if (!menuToggle || !nav) return;

        // Create overlay element for mobile menu backdrop
        const overlay = document.createElement('div');
        overlay.className = 'nav-overlay';
        document.body.appendChild(overlay);

        var scrollY = 0;

        function openMenu() {
            scrollY = window.pageYOffset;
            nav.classList.add('active');
            menuToggle.classList.add('active');
            overlay.classList.add('active');
            body.classList.add('nav-open');
            body.style.top = '-' + scrollY + 'px';
            menuToggle.setAttribute('aria-expanded', 'true');
        }

        function closeMenu() {
            nav.classList.remove('active');
            menuToggle.classList.remove('active');
            overlay.classList.remove('active');
            body.classList.remove('nav-open');
            body.style.top = '';
            window.scrollTo(0, scrollY);
            menuToggle.setAttribute('aria-expanded', 'false');
        }

        function isMenuOpen() {
            return nav.classList.contains('active');
        }

        menuToggle.addEventListener('click', function(e) {
            e.stopPropagation();
            if (isMenuOpen()) {
                closeMenu();
            } else {
                openMenu();
            }
        });

        // Close on overlay click
        overlay.addEventListener('click', closeMenu);

        // Close on escape
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape' && isMenuOpen()) {
                closeMenu();
            }
        });

        // Close when clicking outside
        document.addEventListener('click', function(e) {
            if (isMenuOpen() &&
                !nav.contains(e.target) &&
                !menuToggle.contains(e.target)) {
                closeMenu();
            }
        });

        // Close menu when a nav link is clicked (mobile)
        nav.querySelectorAll('.nav-link').forEach(function(link) {
            link.addEventListener('click', function() {
                if (window.innerWidth <= 768) {
                    closeMenu();
                }
            });
        });
    }

    // ========================================
    // Mobile Sidebar (Tutorial pages)
    // ========================================
    function initMobileSidebar() {
        var sidebar = document.querySelector('.tutorial-sidebar');
        var main = document.querySelector('.tutorial-main');
        if (!sidebar || !main) return;

        // Create toggle button
        var toggleBtn = document.createElement('button');
        toggleBtn.className = 'sidebar-mobile-toggle';
        toggleBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12h18M3 6h18M3 18h18"/></svg> <span>Navigate Tutorials</span>';
        main.insertBefore(toggleBtn, main.firstChild);

        // Create overlay
        var overlay = document.createElement('div');
        overlay.className = 'sidebar-overlay';
        document.body.appendChild(overlay);

        function openSidebar() {
            sidebar.classList.add('mobile-open');
            overlay.classList.add('active');
        }

        function closeSidebar() {
            sidebar.classList.remove('mobile-open');
            overlay.classList.remove('active');
        }

        toggleBtn.addEventListener('click', openSidebar);
        overlay.addEventListener('click', closeSidebar);

        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape' && sidebar.classList.contains('mobile-open')) {
                closeSidebar();
            }
        });

        // Close sidebar when a link is clicked
        sidebar.querySelectorAll('.sidebar-link').forEach(function(link) {
            link.addEventListener('click', function() {
                if (window.innerWidth <= 1024) {
                    closeSidebar();
                }
            });
        });
    }

    // ========================================
    // Scroll Effects
    // ========================================
    function initScrollEffects() {
        const header = document.querySelector('.navbar');
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
                    const headerHeight = document.querySelector('.navbar')?.offsetHeight || 0;
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
