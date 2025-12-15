/**
 * WikiSearch Frontend Application
 * Handles search functionality and result display
 */

// DOM Elements
const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const resultsDiv = document.getElementById('results');

// Configuration
const API_BASE_URL = '';  // Empty string for same origin
const WIKIPEDIA_BASE_URL = 'https://en.wikipedia.org/?curid=';

/**
 * Perform search query
 */
function performSearch() {
    const query = searchInput.value.trim();
    
    if (!query) {
        showMessage('Please enter a search term', 'no-results');
        return;
    }
    
    showMessage('⏳ Searching...', 'loading');
    
    fetch(`${API_BASE_URL}/search?query=${encodeURIComponent(query)}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            displayResults(data);
        })
        .catch(error => {
            showMessage('⚠️ Error performing search. Please try again.', 'no-results');
            console.error('Search error:', error);
        });
}

/**
 * Display search results
 * @param {Array} results - Array of [id, title] tuples
 */
function displayResults(results) {
    if (!results || results.length === 0) {
        showMessage('No results found. Try a different search term.', 'no-results');
        return;
    }
    
    const resultsHTML = results.map(([id, title]) => createResultItem(id, title)).join('');
    resultsDiv.innerHTML = resultsHTML;
}

/**
 * Create HTML for a single result item
 * @param {number} id - Wikipedia article ID
 * @param {string} title - Article title
 * @returns {string} HTML string
 */
function createResultItem(id, title) {
    const escapedTitle = escapeHtml(title);
    return `
        <div class="result-item" onclick="openWikipediaArticle(${id})">
            <div class="result-title">${escapedTitle}</div>
            <div class="result-id">Article ID: ${id}</div>
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
function openWikipediaArticle(id) {
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
