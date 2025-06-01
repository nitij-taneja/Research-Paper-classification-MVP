// Main JavaScript for Research Papers Classification
document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const selectedFile = document.getElementById('selected-file');
    const fileName = document.getElementById('file-name');
    const fileSize = document.getElementById('file-size');
    const removeFile = document.getElementById('remove-file');
    const uploadBtn = document.getElementById('upload-btn');
    const uploadSection = document.getElementById('upload-section');
    const processingSection = document.getElementById('processing-section');
    const resultsSection = document.getElementById('results-section');
    const progressBar = document.getElementById('progress-bar');
    const processingMessage = document.getElementById('processing-message');
    const analyzeAnotherBtn = document.getElementById('analyze-another-btn');
    
    // File selection
    let selectedFileData = null;
    
    // Drag and drop functionality
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        dropArea.classList.add('dragover');
    }
    
    function unhighlight() {
        dropArea.classList.remove('dragover');
    }
    
    dropArea.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            handleFiles(files[0]);
        }
    }
    
    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            handleFiles(this.files[0]);
        }
    });
    
    function handleFiles(file) {
        // Check if file is PDF
        if (file.type !== 'application/pdf') {
            alert('Please upload a PDF file.');
            return;
        }
        
        // Check file size (max 16MB)
        if (file.size > 16 * 1024 * 1024) {
            alert('File size exceeds 16MB limit.');
            return;
        }
        
        selectedFileData = file;
        
        // Display file info
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        selectedFile.classList.remove('d-none');
        uploadBtn.disabled = false;
        
        // Add animation
        selectedFile.classList.add('fade-in');
        setTimeout(() => {
            selectedFile.classList.remove('fade-in');
        }, 500);
    }
    
    function formatFileSize(bytes) {
        if (bytes < 1024) {
            return bytes + ' bytes';
        } else if (bytes < 1024 * 1024) {
            return (bytes / 1024).toFixed(1) + ' KB';
        } else {
            return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
        }
    }
    
    // Remove selected file
    removeFile.addEventListener('click', function() {
        selectedFileData = null;
        selectedFile.classList.add('d-none');
        fileInput.value = '';
        uploadBtn.disabled = true;
    });
    
    // Upload file
    uploadBtn.addEventListener('click', function() {
        if (!selectedFileData) {
            return;
        }
        
        const formData = new FormData();
        formData.append('file', selectedFileData);
        
        // Show processing section
        uploadSection.classList.add('d-none');
        processingSection.classList.remove('d-none');
        
        // Upload file
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            const submissionId = data.submission_id;
            checkStatus(submissionId);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error uploading file: ' + error.message);
            // Show upload section again
            processingSection.classList.add('d-none');
            uploadSection.classList.remove('d-none');
        });
    });
    
    // Check processing status
    function checkStatus(submissionId) {
        fetch(`/status/${submissionId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Update progress
            progressBar.style.width = data.progress + '%';
            processingMessage.textContent = data.message;
            
            if (data.status === 'completed') {
                // Show results
                setTimeout(() => {
                    processingSection.classList.add('d-none');
                    resultsSection.classList.remove('d-none');
                    displayResults(data.result);
                    loadVisualizations(submissionId);
                }, 1000);
            } else if (data.status === 'error') {
                // Show error
                alert('Error processing file: ' + data.message);
                processingSection.classList.add('d-none');
                uploadSection.classList.remove('d-none');
            } else {
                // Continue checking
                setTimeout(() => {
                    checkStatus(submissionId);
                }, 2000);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error checking status: ' + error.message);
            processingSection.classList.add('d-none');
            uploadSection.classList.remove('d-none');
        });
    }
    
    // Display results
    function displayResults(result) {
        const resultIconContainer = document.getElementById('result-icon-container');
        const resultTitle = document.getElementById('result-title');
        const resultSummary = document.getElementById('result-summary');
        const confidenceBar = document.getElementById('confidence-bar');
        const confidenceValue = document.getElementById('confidence-value');
        const publishableContent = document.getElementById('publishable-content');
        const recommendationCard = document.getElementById('recommendation-card');
        
        // Set confidence
        const confidence = Math.round(result.confidence * 100);
        confidenceBar.style.width = confidence + '%';
        confidenceValue.textContent = confidence + '%';
        
        // Set color based on confidence
        if (confidence >= 80) {
            confidenceBar.classList.add('bg-success');
        } else if (confidence >= 50) {
            confidenceBar.classList.add('bg-warning');
        } else {
            confidenceBar.classList.add('bg-danger');
        }
        
        // Set result icon and title
        if (result.publishable) {
            resultIconContainer.innerHTML = '<i class="fas fa-check-circle text-success fa-4x"></i>';
            resultIconContainer.classList.add('publishable-icon');
            resultTitle.textContent = 'Publishable Paper';
            resultSummary.textContent = 'Your paper has been assessed as publishable with ' + confidence + '% confidence.';
            
            // Set publishable content
            publishableContent.innerHTML = `
                <div class="alert alert-success">
                    <i class="fas fa-check-circle me-2"></i>
                    <strong>Congratulations!</strong> Your paper meets the criteria for publication.
                </div>
                <p>Based on our analysis, your paper demonstrates strong methodology, clear results, and valuable contributions to the field.</p>
            `;
            
            // Show recommendation card
            recommendationCard.classList.remove('d-none');
            document.getElementById('conference-name').textContent = result.conference;
            document.getElementById('rationale-text').textContent = result.rationale;
        } else {
            resultIconContainer.innerHTML = '<i class="fas fa-times-circle text-danger fa-4x"></i>';
            resultIconContainer.classList.add('non-publishable-icon');
            resultTitle.textContent = 'Needs Improvement';
            resultSummary.textContent = 'Your paper may need revisions before submission.';
            
            // Set non-publishable content
            publishableContent.innerHTML = `
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    <strong>Improvement Needed</strong> Your paper may need revisions before submission.
                </div>
                <p>Based on our analysis, your paper may benefit from improvements in methodology, clarity, or contributions to the field.</p>
                <div class="card mt-3">
                    <div class="card-header bg-light">
                        <h5 class="mb-0">Improvement Suggestions</h5>
                    </div>
                    <div class="card-body">
                        <ul class="mb-0">
                            <li>Strengthen your methodology section with more details</li>
                            <li>Provide more comprehensive results and analysis</li>
                            <li>Clarify your research contributions and significance</li>
                            <li>Improve the structure and flow of your paper</li>
                            <li>Enhance your literature review with more recent references</li>
                        </ul>
                    </div>
                </div>
            `;
            
            // Hide recommendation card
            recommendationCard.classList.add('d-none');
        }
        
        // Add animation
        resultsSection.classList.add('fade-in');
    }
    
    // Load visualizations
    function loadVisualizations(submissionId) {
        fetch(`/visualizations/${submissionId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(visualizations => {
            const visualizationContainer = document.getElementById('visualization-container');
            
            if (visualizations.length === 0) {
                visualizationContainer.innerHTML = `
                    <div class="col-12 text-center py-4">
                        <p>No visualizations available for this analysis.</p>
                    </div>
                `;
                return;
            }
            
            let html = '';
            visualizations.forEach(vis => {
                html += `
                    <div class="col-md-6 mb-4">
                        <div class="card h-100">
                            <div class="card-header bg-light">
                                <h5 class="mb-0">${vis.name}</h5>
                            </div>
                            <div class="card-body text-center">
                                <img src="${vis.path}" alt="${vis.name}" class="img-fluid">
                            </div>
                        </div>
                    </div>
                `;
            });
            
            visualizationContainer.innerHTML = html;
        })
        .catch(error => {
            console.error('Error:', error);
            const visualizationContainer = document.getElementById('visualization-container');
            visualizationContainer.innerHTML = `
                <div class="col-12 text-center py-4">
                    <p class="text-danger">Error loading visualizations: ${error.message}</p>
                </div>
            `;
        });
    }
    
    // Analyze another paper
    analyzeAnotherBtn.addEventListener('click', function() {
        // Reset form
        selectedFileData = null;
        selectedFile.classList.add('d-none');
        fileInput.value = '';
        uploadBtn.disabled = true;
        
        // Show upload section
        resultsSection.classList.add('d-none');
        uploadSection.classList.remove('d-none');
        
        // Reset progress
        progressBar.style.width = '0%';
        progressBar.classList.remove('bg-success', 'bg-warning', 'bg-danger');
        progressBar.classList.add('bg-primary');
        
        // Reset result icon
        const resultIconContainer = document.getElementById('result-icon-container');
        resultIconContainer.classList.remove('publishable-icon', 'non-publishable-icon');
        
        // Reset confidence bar
        const confidenceBar = document.getElementById('confidence-bar');
        confidenceBar.style.width = '0%';
        confidenceBar.classList.remove('bg-success', 'bg-warning', 'bg-danger');
    });
});
