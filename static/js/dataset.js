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
     * Note: This functionality has been removed as requested
     */
    function setupSyntheticDataHandlers() {
        // Synthetic data generation functionality has been removed
        console.debug('Synthetic data generation functionality has been removed');
        return;
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
        const numSamplesInfo = document.getElementById('numSamplesInfo');
        const numFeaturesInfo = document.getElementById('numFeaturesInfo');
        const targetNameInfo = document.getElementById('targetNameInfo');
        const featuresList = document.getElementById('featuresList');
        
        const datasetInfoObj = data.dataset_info || data;
        
        if (numSamplesInfo) {
            numSamplesInfo.textContent = datasetInfoObj.num_samples || 0;
        }
        
        if (numFeaturesInfo) {
            numFeaturesInfo.textContent = datasetInfoObj.num_features || 0;
        }
        
        if (targetNameInfo) {
            targetNameInfo.textContent = datasetInfoObj.target_column || '-';
        }
        
        // Update features list if available
        if (featuresList && datasetInfoObj.feature_names && Array.isArray(datasetInfoObj.feature_names)) {
            // Clear existing features
            featuresList.innerHTML = '';
            
            // Add each feature as a list item
            datasetInfoObj.feature_names.forEach(feature => {
                const li = document.createElement('li');
                li.className = 'list-group-item';
                li.textContent = feature;
                featuresList.appendChild(li);
            });
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
