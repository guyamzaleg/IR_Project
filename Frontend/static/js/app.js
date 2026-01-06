
/**
 * G&M SEARCH MOTORS
 * Exclusive Search Experience
 * Engineered for Excellence
 */

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const searchInput = document.getElementById('searchInput');
    const searchBtn = document.getElementById('searchBtn');
    const resultsDiv = document.getElementById('results');
    const showPageRankCheckbox = document.getElementById('showPageRank');
    const showPageViewsCheckbox = document.getElementById('showPageViews');

    // Configuration
    const API_BASE_URL = '';  // Empty string for same origin
    const WIKIPEDIA_BASE_URL = 'https://en.wikipedia.org/?curid=';
    
    // Store current results for metrics fetching
    let currentResults = [];

    /**
     * Get selected search mode
     */
    function getSearchMode() {
        const selectedMode = document.querySelector('input[name="searchMode"]:checked');
        return selectedMode ? selectedMode.value : 'search';
    }

    /**
     * Perform search query
     */
    function performSearch() {
        const query = searchInput.value.trim();
        
        if (!query) {
            showMessage('PLEASE ENTER A SEARCH QUERY', 'no-results');
            return;
        }
        
        const searchMode = getSearchMode();
        const modeNames = {
            'search': 'HYBRID',
            'search_body': 'BODY',
            'search_title': 'TITLE',
            'search_anchor': 'ANCHOR'
        };
        
        showMessage(`SEARCHING WIKIPEDIA (${modeNames[searchMode]} MODE)...`, 'loading');
        
        fetch(`${API_BASE_URL}/${searchMode}?query=${encodeURIComponent(query)}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.error) {
                    showMessage(`ERROR: ${data.error.toUpperCase()}`, 'no-results');
                } else {
                    currentResults = data;
                    displayResults(data);
                    fetchMetrics(data);
                }
            })
            .catch(error => {
                showMessage('SEARCH ENGINE ERROR. PLEASE TRY AGAIN.', 'no-results');
                console.error('Search error:', error);
            });
    }

    /**
     * Fetch PageRank and PageView metrics for results
     */
    function fetchMetrics(results) {
        if (!results || results.length === 0) return;
        
        const docIds = results.map(r => Array.isArray(r) ? r[0] : r);
        
        // Fetch PageRank if enabled
        if (showPageRankCheckbox.checked) {
            fetch(`${API_BASE_URL}/get_pagerank`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(docIds)
            })
            .then(response => response.json())
            .then(pageranks => {
                if (!pageranks.error) {
                    updateMetrics('pagerank', docIds, pageranks);
                }
            })
            .catch(error => console.error('PageRank fetch error:', error));
        }
        
        // Fetch PageViews if enabled
        if (showPageViewsCheckbox.checked) {
            fetch(`${API_BASE_URL}/get_pageview`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(docIds)
            })
            .then(response => response.json())
            .then(pageviews => {
                if (!pageviews.error) {
                    updateMetrics('pageview', docIds, pageviews);
                }
            })
            .catch(error => console.error('PageView fetch error:', error));
        }
    }

    /**
     * Update result items with metrics
     */
    function updateMetrics(type, docIds, values) {
        docIds.forEach((docId, index) => {
            const resultItem = document.querySelector(`[data-doc-id="${docId}"]`);
            if (resultItem && values[index] !== undefined) {
                const metricsDiv = resultItem.querySelector('.result-metrics');
                if (metricsDiv) {
                    const metricSpan = document.createElement('span');
                    metricSpan.className = `metric-${type}`;
                    
                    if (type === 'pagerank') {
                        metricSpan.innerHTML = `<strong>PR:</strong> ${values[index].toFixed(6)}`;
                    } else if (type === 'pageview') {
                        metricSpan.innerHTML = `<strong>Views:</strong> ${values[index].toLocaleString()}`;
                    }
                    
                    metricsDiv.appendChild(metricSpan);
                }
            }
        });
    }

    /**
     * Display search results
     * @param {Array} results - Array of [id, title] tuples
     */
    function displayResults(results) {
        if (!results || results.length === 0) {
            showMessage('NO RESULTS FOUND. TRY A DIFFERENT QUERY.', 'no-results');
            return;
        }
        
        const resultsHeader = `<div class="results-header">${results.length} RESULT${results.length !== 1 ? 'S' : ''} FOUND</div>`;
        const resultsHTML = results.map(([id, title], index) => createResultItem(id, title, index + 1)).join('');
        resultsDiv.innerHTML = resultsHeader + resultsHTML;
    }

    /**
     * Create HTML for a single result item
     * @param {number} id - Wikipedia article ID
     * @param {string} title - Article title
     * @param {number} index - Result number
     * @returns {string} HTML string
     */
    function createResultItem(id, title, index) {
        const escapedTitle = escapeHtml(title);
        return `
            <div class="result-item" data-doc-id="${id}" onclick="openWikipediaArticle(${id})">
                <div class="result-number">#${index.toString().padStart(2, '0')}</div>
                <div class="result-title">${escapedTitle}</div>
                <div class="result-id">DOCUMENT ID: ${id}</div>
                <div class="result-metrics"></div>
                <div class="result-link">en.wikipedia.org</div>
            </div>
        `;
    }

    /**
     * Show a message in the results area
     * @param {string} message - Message to display
     * @param {string} className - CSS class for styling
     */
    function showMessage(message, className) {
        resultsDiv.innerHTML = `<div class="${className}">${message}</div>`;
    }

    /**
     * Open Wikipedia article in new tab
     * @param {number} id - Wikipedia article ID
     */
    window.openWikipediaArticle = function(id) {
        window.open(`${WIKIPEDIA_BASE_URL}${id}`, '_blank');
    }

    /**
     * Escape HTML to prevent XSS
     * @param {string} text - Text to escape
     * @returns {string} Escaped text
     */
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Handle Enter key press in search input
     * @param {KeyboardEvent} event
     */
    function handleKeyPress(event) {
        if (event.key === 'Enter') {
            performSearch();
        }
    }

    // Event Listeners
    searchBtn.addEventListener('click', performSearch);
    searchInput.addEventListener('keypress', handleKeyPress);

    // Focus on input when page loads
    searchInput.focus();

}); // End of DOMContentLoaded
