/**
 * Neural Network Optimization Project
 * Main JavaScript File
 * 
 * Uses module pattern for better organization and to avoid global variables
 */

// Main application module
const OptimizationApp = (function() {
    // Private variables
    let _gaParams = null;
    let _psoParams = null;
    let _hasUnsavedChanges = false;

    /**
     * Initialize the application
     */
    function init() {
        // Initialize tooltips
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function(tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });

        // Show confirmation before leaving page with unsaved changes
        window.addEventListener('beforeunload', function(e) {
            // Check if there are any unsaved changes
            if (_hasUnsavedChanges) {
                e.preventDefault();
                e.returnValue = '';
                return '';
            }
        });
        
        // Set up event handlers
        setupEventHandlers();
        
        // Initialize from URL parameters
        initFromUrlParams();
    }

    /**
     * Set up all event handlers
     */
    function setupEventHandlers() {
        // Apply GA configuration - only on algorithms page
        const applyGaConfigBtn = document.getElementById('applyGaConfig');
        if (applyGaConfigBtn) {
            applyGaConfigBtn.addEventListener('click', function() {
                // Collect form data
                const gaForm = document.getElementById('gaForm');
                if (!gaForm) return;
                
                const formData = new FormData(gaForm);
                
                // Convert value range to array
                const valueRangeMin = parseFloat(formData.get('value_range_min'));
                const valueRangeMax = parseFloat(formData.get('value_range_max'));
                
                // Create config object
                _gaParams = {
                    population_size: parseInt(formData.get('population_size')),
                    num_generations: parseInt(formData.get('num_generations')),
                    chromosome_type: formData.get('chromosome_type'),
                    value_range: [valueRangeMin, valueRangeMax],
                    selection_method: formData.get('selection_method'),
                    tournament_size: parseInt(formData.get('tournament_size')),
                    crossover_rate: parseFloat(formData.get('crossover_rate')),
                    mutation_rate: parseFloat(formData.get('mutation_rate')),
                    elitism: formData.get('elitism') === 'on',
                    elite_size: parseInt(formData.get('elite_size')),
                    crossover_type: formData.get('crossover_type'),
                    adaptive_mutation: formData.get('adaptive_mutation') === 'on'
                };
                
                _hasUnsavedChanges = true;
                OptimizationApp.showToast('GA configuration applied successfully!');
            });
        }
        
        // Save GA configuration - only on algorithms page
        const saveGaConfigBtn = document.getElementById('saveGaConfig');
        if (saveGaConfigBtn) {
            saveGaConfigBtn.addEventListener('click', function() {
                // Get config name from user
                const configName = prompt('Enter a name for this GA configuration:');
                if (!configName) return;
                
                const gaForm = document.getElementById('gaForm');
                if (!gaForm) return;
                
                const formData = new FormData(gaForm);
                
                // Convert value range to array
                const valueRangeMin = parseFloat(formData.get('value_range_min'));
                const valueRangeMax = parseFloat(formData.get('value_range_max'));
                
                // Create config object
                _gaParams = {
                    population_size: parseInt(formData.get('population_size')),
                    num_generations: parseInt(formData.get('num_generations')),
                    chromosome_type: formData.get('chromosome_type'),
                    value_range: [valueRangeMin, valueRangeMax],
                    selection_method: formData.get('selection_method'),
                    tournament_size: parseInt(formData.get('tournament_size')),
                    crossover_rate: parseFloat(formData.get('crossover_rate')),
                    mutation_rate: parseFloat(formData.get('mutation_rate')),
                    elitism: formData.get('elitism') === 'true'
                };
                
                // Show success message
                showToast('GA configuration saved successfully', 'success');
                
                // Mark as having unsaved changes
                _hasUnsavedChanges = true;
            });
        }
        
        // Apply PSO configuration - only on algorithms page
        const applyPsoConfigBtn = document.getElementById('applyPsoConfig');
        if (applyPsoConfigBtn) {
            applyPsoConfigBtn.addEventListener('click', function() {
                // Collect form data
                const psoForm = document.getElementById('psoForm');
                if (!psoForm) return;
                
                const formData = new FormData(psoForm);
            
                // Convert value range to array
                const valueRangeMin = parseFloat(formData.get('value_range_min'));
                const valueRangeMax = parseFloat(formData.get('value_range_max'));
                
                // Create config object
                _psoParams = {
                    num_particles: parseInt(formData.get('num_particles')),
                    num_iterations: parseInt(formData.get('num_iterations')),
                    value_range: [valueRangeMin, valueRangeMax],
                    discrete: formData.get('discrete') === 'on',
                    inertia_weight: parseFloat(formData.get('inertia_weight')),
                    cognitive_coefficient: parseFloat(formData.get('cognitive_coefficient')),
                    social_coefficient: parseFloat(formData.get('social_coefficient')),
                    max_velocity: parseFloat(formData.get('max_velocity')),
                    decreasing_inertia: formData.get('decreasing_inertia') === 'on',
                    final_inertia_weight: parseFloat(formData.get('final_inertia_weight')),
                    use_neighborhood: formData.get('use_neighborhood') === 'on',
                    neighborhood_size: parseInt(formData.get('neighborhood_size'))
                };
                
                _hasUnsavedChanges = true;
                showToast('PSO configuration applied successfully!');
            });
        }
        
        // Save PSO configuration - only on algorithms page
        const savePsoConfigBtn = document.getElementById('savePsoConfig');
        if (savePsoConfigBtn) {
            savePsoConfigBtn.addEventListener('click', function() {
                // Get config name from user
                const configName = prompt('Enter a name for this PSO configuration:');
                if (!configName) return;
                
                const psoForm = document.getElementById('psoForm');
                if (!psoForm) return;
                
                const formData = new FormData(psoForm);
                const valueRangeMin = parseFloat(formData.get('value_range_min'));
                const valueRangeMax = parseFloat(formData.get('value_range_max'));
                
                // Create config object
                _psoParams = {
                    num_particles: parseInt(formData.get('num_particles')),
                    num_iterations: parseInt(formData.get('num_iterations')),
                    value_range: [valueRangeMin, valueRangeMax],
                    discrete: formData.get('discrete') === 'on',
                    inertia_weight: parseFloat(formData.get('inertia_weight')),
                    cognitive_coefficient: parseFloat(formData.get('cognitive_coefficient')),
                    social_coefficient: parseFloat(formData.get('social_coefficient')),
                    max_velocity: parseFloat(formData.get('max_velocity')),
                    decreasing_inertia: formData.get('decreasing_inertia') === 'on',
                    final_inertia_weight: parseFloat(formData.get('final_inertia_weight')),
                    use_neighborhood: formData.get('use_neighborhood') === 'on',
                    neighborhood_size: parseInt(formData.get('neighborhood_size'))
                };
                
                _hasUnsavedChanges = true;
                showToast('PSO configuration applied successfully!');
            });
        }
    } // <-- Missing closing brace for setupEventHandlers function

    /**
     * Show toast notification
     * @param {string} message - Message to display
     * @param {string} type - Type of toast (success, danger, etc.)
     */
    function showToast(message, type = 'success') {
        // Create toast container if it doesn't exist
        let toastContainer = document.getElementById('toastContainer');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.id = 'toastContainer';
            toastContainer.className = 'position-fixed bottom-0 end-0 p-3';
            toastContainer.style.zIndex = '5';
            document.body.appendChild(toastContainer);
        }
        
        // Create unique ID for this toast
        const toastId = 'toast-' + Date.now();
        
        // Create toast element
        const toastDiv = document.createElement('div');
        toastDiv.id = toastId;
        toastDiv.className = 'toast';
        toastDiv.setAttribute('role', 'alert');
        toastDiv.setAttribute('aria-live', 'assertive');
        toastDiv.setAttribute('aria-atomic', 'true');
        
        // Create toast header
        const headerDiv = document.createElement('div');
        headerDiv.className = `toast-header bg-${type} text-white`;
        
        const strongEl = document.createElement('strong');
        strongEl.className = 'me-auto';
        strongEl.textContent = type === 'success' ? 'Success' : 'Error';
        
        const closeButton = document.createElement('button');
        closeButton.type = 'button';
        closeButton.className = 'btn-close btn-close-white';
        closeButton.setAttribute('data-bs-dismiss', 'toast');
        closeButton.setAttribute('aria-label', 'Close');
        
        headerDiv.appendChild(strongEl);
        headerDiv.appendChild(closeButton);
        
        // Create toast body
        const bodyDiv = document.createElement('div');
        bodyDiv.className = 'toast-body';
        bodyDiv.textContent = message;
        
        // Assemble toast
        toastDiv.appendChild(headerDiv);
        toastDiv.appendChild(bodyDiv);
        
        // Add toast to container
        toastContainer.appendChild(toastDiv);
        
        // Initialize and show the toast
        const toastElement = new bootstrap.Toast(toastDiv, {
            autohide: true,
            delay: 5000
        });
        toastElement.show();
        
        // Remove toast from DOM when hidden
        toastDiv.addEventListener('hidden.bs.toast', function() {
            toastDiv.remove();
        });
    }

    /**
     * Format numbers with commas
     * @param {number} num - Number to format
     * @returns {string} Formatted number
     */
    function formatNumber(num) {
        return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    }

    /**
     * Create a chart
     * @param {string} canvasId - Canvas element ID
     * @param {string} type - Chart type
     * @param {Object} data - Chart data
     * @param {Object} options - Chart options
     * @returns {Chart} Chart instance
     */
    function createChart(canvasId, type, data, options) {
        const ctx = document.getElementById(canvasId).getContext('2d');
        return new Chart(ctx, {
            type: type,
            data: data,
            options: options
        });
    }

    /**
     * Download data as a file
     * @param {string} data - Data to download
     * @param {string} filename - File name
     * @param {string} type - MIME type
     */
    function downloadData(data, filename, type) {
        const blob = new Blob([data], { type: type });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    }

    /**
     * Validate form inputs
     * @param {string} formId - Form element ID
     * @returns {boolean} Form validity
     */
    function validateForm(formId) {
        const form = document.getElementById(formId);
        if (!form) return true;
        
        return form.checkValidity();
    }

    /**
     * Show loading spinner
     * @param {string} elementId - Element ID
     * @param {string} message - Loading message
     */
    function showLoading(elementId, message = 'Loading...') {
        const element = document.getElementById(elementId);
        if (!element) return;
        
        const originalContent = element.innerHTML;
        element.setAttribute('data-original-content', originalContent);
        
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'd-flex justify-content-center align-items-center';
        
        const spinnerDiv = document.createElement('div');
        spinnerDiv.className = 'spinner-border spinner-border-sm me-2';
        spinnerDiv.setAttribute('role', 'status');
        
        const spinnerSpan = document.createElement('span');
        spinnerSpan.className = 'visually-hidden';
        spinnerSpan.textContent = 'Loading...';
        
        const messageSpan = document.createElement('span');
        messageSpan.textContent = message;
        
        spinnerDiv.appendChild(spinnerSpan);
        loadingDiv.appendChild(spinnerDiv);
        loadingDiv.appendChild(messageSpan);
        
        element.innerHTML = '';
        element.appendChild(loadingDiv);
    }

    /**
     * Hide loading spinner
     * @param {string} elementId - Element ID
     */
    function hideLoading(elementId) {
        const element = document.getElementById(elementId);
        if (!element) return;
        
        const originalContent = element.getAttribute('data-original-content');
        if (originalContent) {
            element.innerHTML = originalContent;
            element.removeAttribute('data-original-content');
        }
    }

    /**
     * Handle AJAX errors
     * @param {Object} xhr - XMLHttpRequest object
     * @param {string} status - Status text
     * @param {string} error - Error message
     */
    function handleAjaxError(xhr, status, error) {
        console.error('AJAX Error:', status, error);
        
        let errorMessage = 'An error occurred while processing your request.';
        
        if (xhr.responseJSON) {
            if (xhr.responseJSON.error) {
                errorMessage = xhr.responseJSON.error;
            } else if (xhr.responseJSON.message) {
                errorMessage = xhr.responseJSON.message;
            }
        }
        
        // Use the OptimizationApp namespace to call showToast
        OptimizationApp.showToast(errorMessage, 'danger');
    }

    /**
     * Update URL parameters without reloading the page
     * @param {Object} params - Parameters to update
     */
    function updateUrlParams(params) {
        const url = new URL(window.location.href);
        
        for (const key in params) {
            if (params[key] === null) {
                url.searchParams.delete(key);
            } else {
                url.searchParams.set(key, params[key]);
            }
        }
        
        window.history.replaceState({}, '', url);
    }

    /**
     * Get URL parameters
     * @returns {Object} URL parameters
     */
    function getUrlParams() {
        const params = {};
        const searchParams = new URLSearchParams(window.location.search);
        
        for (const [key, value] of searchParams.entries()) {
            params[key] = value;
        }
        
        return params;
    }

    /**
     * Initialize based on URL parameters
     */
    function initFromUrlParams() {
        const params = getUrlParams();
        
        // Handle experiment type
        if (params.experiment_type) {
            const radioInput = document.querySelector(`input[name="experiment_type"][value="${params.experiment_type}"]`);
            if (radioInput) {
                radioInput.checked = true;
                // Create and dispatch a change event
                const event = new Event('change');
                radioInput.dispatchEvent(event);
            }
        }
        
        // Handle algorithm
        if (params.algorithm) {
            const radioInput = document.querySelector(`input[name="algorithm"][value="${params.algorithm}"]`);
            if (radioInput) {
                radioInput.checked = true;
            }
        }
    }

    // Public API
    return {
        init: init,
        showToast: showToast,
        formatNumber: formatNumber,
        createChart: createChart,
        downloadData: downloadData,
        validateForm: validateForm,
        showLoading: showLoading,
        hideLoading: hideLoading,
        handleAjaxError: handleAjaxError,
        updateUrlParams: updateUrlParams,
        getUrlParams: getUrlParams,
        getGaParams: function() { return _gaParams; },
        getPsoParams: function() { return _psoParams; },
        setHasUnsavedChanges: function(value) { _hasUnsavedChanges = value; }
    };
})();

// Initialize the application when the DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    OptimizationApp.init();
});