/**
 * JavaScript for experiment setup and execution
 * Uses module pattern for better organization and code quality
 */

const ExperimentsModule = (function() {
    // Private variables
    let _currentExperiment = null;
    let _experimentRunning = false;

    /**
     * Initialize the module
     */
    function init() {
        // Set up event handlers
        setupEventHandlers();

        // Trigger the initial experiment type change to show the correct parameters
        const selectedExperimentType = document.querySelector('input[name="experiment_type"]:checked');
        if (selectedExperimentType) {
            // Create and dispatch a change event
            const event = new Event('change');
            selectedExperimentType.dispatchEvent(event);
        }
    }

    /**
     * Set up all event handlers
     */
    function setupEventHandlers() {
        // Show/hide parameters based on experiment type
        const experimentTypeInputs = document.querySelectorAll('input[name="experiment_type"]');
        experimentTypeInputs.forEach(input => {
            input.addEventListener('change', function() {
                const experimentType = this.value;

                // Hide all parameter sections
                const paramSections = document.querySelectorAll('.experiment-params');
                paramSections.forEach(section => {
                    section.style.display = 'none';
                });

                // Show relevant parameters
                const relevantSection = document.getElementById(`${experimentType}Params`);
                if (relevantSection) {
                    relevantSection.style.display = 'block';
                }

                // Update URL parameter
                OptimizationApp.updateUrlParams({
                    experiment_type: experimentType
                });
            });
        });

        // Handle algorithm selection
        const algorithmInputs = document.querySelectorAll('input[name="algorithm"]');
        algorithmInputs.forEach(input => {
            input.addEventListener('change', function() {
                const algorithm = this.value;

                // Update URL parameter
                OptimizationApp.updateUrlParams({
                    algorithm: algorithm
                });
            });
        });

        // Handle experiment form submission
        const experimentForm = document.getElementById('experimentForm');
        if (experimentForm) {
            experimentForm.addEventListener('submit', runExperiment);
        }

        // Cancel experiment
        const cancelExperiment = document.getElementById('cancelExperiment');
        if (cancelExperiment) {
            cancelExperiment.addEventListener('click', function() {
                if (confirm('Are you sure you want to cancel the experiment?')) {
                    const experimentProgress = document.getElementById('experimentProgress');
                    if (experimentProgress) {
                        experimentProgress.style.display = 'none';
                    }
                    const experimentForm = document.getElementById('experimentForm');
                    if (experimentForm) {
                        experimentForm.style.display = 'block';
                    }
                }
            });
        }
    }

    /**
     * Run the experiment
     * @param {Event} e - Form submit event
     */
    function runExperiment(e) {
        e.preventDefault();

        // Check if experiment is already running
        if (_experimentRunning) {
            return;
        }

        _experimentRunning = true;

        // Get run button and disable it
        const runButton = document.getElementById('runExperimentBtn');
        if (runButton) {
            runButton.disabled = true;
            runButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Running...';
        }

        // Hide the form and show progress section
        const experimentForm = document.getElementById('experimentForm');
        if (experimentForm) {
            experimentForm.style.display = 'none';
        }
        
        // Show progress section with animation
        const progressSection = document.getElementById('experimentProgress');
        if (progressSection) {
            progressSection.style.display = 'block';
        }
        
        // Initialize progress bar animation
        const progressBar = document.getElementById('progressBar');
        if (progressBar) {
            progressBar.style.width = '10%';
            setTimeout(() => { progressBar.style.width = '30%'; }, 500);
        }
        
        // Update status message
        const progressStatus = document.getElementById('progressStatus');
        if (progressStatus) {
            progressStatus.textContent = 'Starting experiment...';
        }

        // Get form data
        const formData = new FormData(e.target);

        // Log form data for debugging
        console.log('Experiment parameters:');
        for (const [key, value] of formData.entries()) {
            console.log(`${key}: ${value}`);
        }

        // Send AJAX request
        fetch('/api/run_experiment', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Server error: ' + response.status);
            }
            
            // Update progress bar to indicate processing
            const progressBar = document.getElementById('progressBar');
            if (progressBar) {
                progressBar.style.width = '70%';
            }
            
            // Update status message
            const progressStatus = document.getElementById('progressStatus');
            if (progressStatus) {
                progressStatus.textContent = 'Processing results...';
            }
            
            return response.json();
        })
        .then(data => {
            if (!data.success) {
                throw new Error(data.error || 'Failed to run experiment');
            }
            return data;
        })
        .then(data => {
            // Handle success
            handleExperimentSuccess(data);
        })
        .catch(error => {
            // Handle error
            handleExperimentError(error);
        })
        .finally(() => {
            // Re-enable run button
            if (runButton) {
                runButton.disabled = false;
                runButton.innerHTML = '<i class="fas fa-play me-2"></i>Run Experiment';
            }
            _experimentRunning = false;
        });
    }

    /**
     * Handle successful experiment run
     * @param {Object} response - Server response
     */
    function handleExperimentSuccess(response) {
        // Show success message
        if (window.OptimizationApp && OptimizationApp.showToast) {
            OptimizationApp.showToast(response.message, 'success');
        } else {
            console.log('Success:', response.message);
        }
        
        // Update progress
        const progressStatus = document.getElementById('progressStatus');
        if (progressStatus) {
            progressStatus.textContent = 'Experiment completed successfully!';
        }
        
        const progressBar = document.getElementById('progressBar');
        if (progressBar) {
            progressBar.style.width = '100%';
            progressBar.classList.add('bg-success');
        }
        
        // Wait a moment to show the completion state before redirecting
        setTimeout(() => {
            // Check if we should redirect to results page
            if (response.data && response.data.redirect_to) {
                window.location.href = response.data.redirect_to;
            } else {
                // Fallback to results page if no specific redirect
                window.location.href = '/results';
            }
        }, 1000);
    }
    
    /**
     * Handle experiment error
     * @param {Error} error - Error object
     */
    function handleExperimentError(error) {
        // Show error message
        if (window.OptimizationApp && OptimizationApp.showToast) {
            OptimizationApp.showToast(error.message, 'danger');
        } else {
            console.error('Error:', error.message);
        }
        
        // Update progress
        const progressStatus = document.getElementById('progressStatus');
        if (progressStatus) {
            progressStatus.textContent = 'Experiment failed: ' + error.message;
        }
        
        const progressBar = document.getElementById('progressBar');
        if (progressBar) {
            progressBar.style.width = '100%';
            progressBar.classList.add('bg-danger');
        }
        
        // Show the form again after a delay
        setTimeout(() => {
            const experimentForm = document.getElementById('experimentForm');
            if (experimentForm) {
                experimentForm.style.display = 'block';
            }
            
            const progressSection = document.getElementById('experimentProgress');
            if (progressSection) {
                progressSection.style.display = 'none';
            }
        }, 3000);
    }
    
    /**
     * Initialize experiment type and algorithm based on URL parameters
     */
    function initFromUrlParams() {
        const params = OptimizationApp.getUrlParams();
        
        // Set experiment type
        const experimentType = params.experiment_type;
        if (experimentType) {
            const radioInput = document.querySelector(`input[name="experiment_type"][value="${experimentType}"]`);
            if (radioInput) {
                radioInput.checked = true;
                // Create and dispatch a change event
                const event = new Event('change');
                radioInput.dispatchEvent(event);
            }
        } else {
            // Trigger the first option by default
            const firstOption = document.querySelector('input[name="experiment_type"]:first-of-type');
            if (firstOption) {
                firstOption.checked = true;
                // Create and dispatch a change event
                const event = new Event('change');
                firstOption.dispatchEvent(event);
            }
        }
        
        // Set algorithm
        const algorithm = params.algorithm;
        if (algorithm) {
            const radioInput = document.querySelector(`input[name="algorithm"][value="${algorithm}"]`);
            if (radioInput) {
                radioInput.checked = true;
            }
        } else {
            // Select 'both' by default
            const bothOption = document.querySelector('input[name="algorithm"][value="both"]');
            if (bothOption) {
                bothOption.checked = true;
            }
        }
    }
    
    // Public API
    return {
        init: function() {
            init();
            initFromUrlParams();
        },
        runExperiment: runExperiment
    };
})();

// Initialize the module when the DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    ExperimentsModule.init();
});
