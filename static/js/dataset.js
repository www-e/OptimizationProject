/**
 * JavaScript for dataset management page
 * Handles file uploads, drag & drop, and synthetic data generation
 * Uses module pattern for better organization and code quality
 */

// Helper function for showing toast notifications
function showToast(message, type = 'success') {
    // Try to use OptimizationApp if available, otherwise create a simple toast
    if (typeof OptimizationApp !== 'undefined' && OptimizationApp.showToast) {
        OptimizationApp.showToast(message, type);
        return;
    }
    
    // Fallback implementation
    const toastContainer = document.getElementById('toastContainer');
    if (!toastContainer) {
        console.error('Toast container not found, message was:', message);
        alert(message); // Fallback to alert if toast container not found
        return;
    }
    
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type} border-0`;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');
    
    const toastBody = document.createElement('div');
    toastBody.className = 'toast-body d-flex align-items-center';
    
    // Add appropriate icon based on type
    let icon = 'info-circle';
    if (type === 'success') icon = 'check-circle';
    if (type === 'danger') icon = 'exclamation-circle';
    if (type === 'warning') icon = 'exclamation-triangle';
    
    toastBody.innerHTML = `<i class="fas fa-${icon} me-2"></i>${message}`;
    
    const closeButton = document.createElement('button');
    closeButton.type = 'button';
    closeButton.className = 'btn-close btn-close-white ms-auto';
    closeButton.setAttribute('data-bs-dismiss', 'toast');
    closeButton.setAttribute('aria-label', 'Close');
    
    toastBody.appendChild(closeButton);
    toast.appendChild(toastBody);
    toastContainer.appendChild(toast);
    
    // Initialize and show the toast
    const bsToast = new bootstrap.Toast(toast, { delay: 5000 });
    bsToast.show();
    
    // Remove toast after it's hidden
    toast.addEventListener('hidden.bs.toast', function() {
        toast.remove();
    });
}

const DatasetModule = (function() {
    // Private variables
    let _uploadInProgress = false;
    let _fileSelected = false;
    
    /**
     * Initialize the module
     */
    function init() {
        console.log('Initializing DatasetModule');
        
        // Set up event handlers
        setupFileUploadHandlers();
        setupSyntheticDataHandlers();
    }
    
    /**
     * Set up file upload event handlers
     */
    function setupFileUploadHandlers() {
        // Get DOM elements
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const uploadForm = document.getElementById('uploadForm');
        const datasetInfo = document.getElementById('datasetInfo');
        
        if (!uploadArea || !fileInput || !uploadForm) {
            console.error('Required DOM elements for file upload not found');
            return;
        }
    
        // Click on upload area to trigger file input label
        uploadArea.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            
            // Find the label for the file input and click it
            const fileInputLabel = document.querySelector('label[for="fileInput"]');
            if (fileInputLabel) {
                fileInputLabel.click();
            }
        });
        
        // Update display when file is selected
        fileInput.addEventListener('change', function() {
            if (this.files && this.files.length > 0) {
                const fileName = this.files[0].name;
                _fileSelected = true;
                
                // Update the upload area text
                const uploadText = uploadArea.querySelector('h5');
                if (uploadText) {
                    uploadText.textContent = 'Selected file: ' + fileName;
                }
                
                // Show upload button
                const uploadButton = document.getElementById('uploadButton');
                if (uploadButton) {
                    uploadButton.style.display = 'block';
                }
            }
        });
        
        // Handle drag and drop events
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, function(e) {
                e.preventDefault();
                e.stopPropagation();
            }, false);
        });
        
        uploadArea.addEventListener('dragenter', function() {
            uploadArea.classList.add('highlight');
        });
        
        uploadArea.addEventListener('dragleave', function() {
            uploadArea.classList.remove('highlight');
        });
        
        uploadArea.addEventListener('drop', function(e) {
            uploadArea.classList.remove('highlight');
            
            // Get the dropped files
            const dt = e.dataTransfer;
            const files = dt.files;
            
            if (files.length > 0) {
                // Update the file input
                fileInput.files = files;
                
                // Trigger change event manually
                const event = new Event('change');
                fileInput.dispatchEvent(event);
            }
        });
        
        // Handle form submission
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Check if a file was selected
            if (!fileInput.files || fileInput.files.length === 0) {
                showToast('Please select a file to upload.', 'danger');
                return;
            }
            
            // Prevent multiple uploads
            if (_uploadInProgress) {
                return;
            }
            
            _uploadInProgress = true;
            
            // Create FormData object
            const formData = new FormData(uploadForm);
            
            // Show loading state
            const uploadButton = document.getElementById('uploadButton');
            if (uploadButton) {
                uploadButton.disabled = true;
                uploadButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Uploading...';
            }
            
            // Send AJAX request
            fetch('/api/upload_dataset', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(data => {
                        throw new Error(data.error || 'Failed to upload dataset');
                    });
                }
                return response.json();
            })
            .then(data => {
                // Reset upload button
                if (uploadButton) {
                    uploadButton.disabled = false;
                    uploadButton.innerHTML = 'Upload';
                    uploadButton.style.display = 'none';
                }
                
                // Reset file input
                fileInput.value = '';
                _fileSelected = false;
                
                // Reset upload area text
                const uploadText = uploadArea.querySelector('h5');
                if (uploadText) {
                    uploadText.textContent = 'Drag & drop your dataset file here or click to browse';
                }
                
                // Show success message
                showToast(data.message || 'Dataset uploaded successfully!', 'success');
                
                // Update dataset info
                updateDatasetInfo(data.data);
            })
            .catch(error => {
                // Reset upload button
                if (uploadButton) {
                    uploadButton.disabled = false;
                    uploadButton.innerHTML = 'Upload';
                }
                
                // Show error message
                showToast('Error: ' + error.message, 'danger');
            })
            .finally(() => {
                _uploadInProgress = false;
            });
        });
    }
    
    /**
     * Set up synthetic data generation event handlers
     */
    function setupSyntheticDataHandlers() {
        // Check if we're on the dataset page by looking for the generateForm element
        const generateForm = document.getElementById('generateForm');
        
        // If we're not on the dataset page, silently return without error
        if (!generateForm) {
            // Only log this as debug info, not an error, since it's expected on other pages
            console.debug('Synthetic data generation form not found (likely not on dataset page)');
            return;
        }
        
        // Handle form submission
        generateForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Get form values
            const numSamplesInput = document.getElementById('numSamples');
            const numFeaturesInput = document.getElementById('numFeatures');
            const noiseInput = document.getElementById('noise');
            
            const numSamples = numSamplesInput ? parseInt(numSamplesInput.value) || 100 : 100;
            const numFeatures = numFeaturesInput ? parseInt(numFeaturesInput.value) || 10 : 10;
            const noise = noiseInput ? parseFloat(noiseInput.value) || 0.1 : 0.1;
            
            // Create request data
            const requestData = {
                num_samples: numSamples,
                num_features: numFeatures,
                noise: noise
            };
            
            // Show loading state
            const submitButton = generateForm.querySelector('button[type="submit"]');
            if (submitButton) {
                submitButton.disabled = true;
                submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Generating...';
            }
            
            // Send request to generate synthetic data
            fetch('/api/generate_dataset', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            })
            .then(response => response.json())
            .then(data => {
                // Reset button state
                if (submitButton) {
                    submitButton.disabled = false;
                    submitButton.innerHTML = '<i class="fas fa-cogs me-2"></i>Generate';
                }
                
                if (data.success) {
                    alert(data.message);
                    updateDatasetInfo(data);
                } else {
                    alert('Error: ' + (data.error || 'Unknown error'));
                }
            })
            .catch(error => {
                console.error('Error:', error);
                
                // Reset button state
                if (submitButton) {
                    submitButton.disabled = false;
                    submitButton.innerHTML = '<i class="fas fa-cogs me-2"></i>Generate';
                }
                
                alert('Error generating dataset: ' + error.message);
            });
        });
    }
    
    /**
     * Update dataset info display
     * @param {Object} data - Dataset information
     */
    function updateDatasetInfo(data) {
        const datasetInfo = document.getElementById('datasetInfo');
        if (!datasetInfo) {
            console.error('Dataset info element not found');
            return;
        }
        
        // Show dataset info section
        datasetInfo.style.display = 'block';
        
        // Update dataset info content
        const numSamples = document.getElementById('numSamples');
        const numFeatures = document.getElementById('numFeatures');
        const featureNames = document.getElementById('featureNames');
        
        const datasetInfoObj = data.dataset_info || data;
        
        if (numSamples) {
            numSamples.textContent = datasetInfoObj.num_samples || 0;
        }
        
        if (numFeatures) {
            numFeatures.textContent = datasetInfoObj.num_features || 0;
        }
        
        if (featureNames && datasetInfoObj.feature_names) {
            featureNames.textContent = datasetInfoObj.feature_names.join(', ');
        }
    }
    
    // Public API
    return {
        init: init
    };
})();

// Initialize the module when the DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    DatasetModule.init();
});
