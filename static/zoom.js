document.addEventListener("DOMContentLoaded", function () {
    const zoomContainer = document.querySelector(".zoom-container");
    const zoomImage = document.querySelector("#MainImg");
    const fabricCircle = document.querySelector("#fabric-circle");
    const details = document.querySelector(".details"); // Select product details

    let zoomLevel = 1; // Initial zoom level
    let detailsVisible = false; // Track if details are visible

    // Handle mouse movement to show fabric details
    zoomContainer.addEventListener('mousemove', function (e) {
        const rect = zoomContainer.getBoundingClientRect(); // Get container size and position

        // Get the mouse position inside the container
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Calculate percentage positions
        const positionXInContainer = x / rect.width * 100;
        const positionYInContainer = y / rect.height * 100;

        zoomImage.style.transformOrigin = `${positionXInContainer}% ${positionYInContainer}%`;
        zoomImage.style.transform = `scale(${zoomLevel})`;

        // Update fabric circle position if it's visible
        if (fabricCircle.style.display === "block") {
            fabricCircle.style.left = e.clientX - 75 + "px"; // Center the circle based on mouse position
            fabricCircle.style.top = e.clientY - 75 + "px";  // Center the circle based on mouse position
        }
    });

    // Hide the circle when the mouse leaves the image
    zoomContainer.addEventListener('mouseleave', function () {
        zoomImage.style.transform = 'scale(1)';
        fabricCircle.style.display = "none"; // Hide the small circle
    });

    // Left-click to toggle zoom and details
    zoomContainer.addEventListener('click', function (e) {
        if (e.button === 0) { // Left mouse button
            zoomLevel += 0.5; // Increase zoom level
            zoomImage.style.transform = `scale(${zoomLevel})`;

            // Show the small circle with fabric details at the click position
            fabricCircle.style.display = "block";
            fabricCircle.style.left = e.clientX - 75 + "px"; // Center the circle based on click position
            fabricCircle.style.top = e.clientY - 75 + "px";  // Center the circle based on click position

            // Toggle details visibility
            if (!detailsVisible) {
                details.style.display = "block"; // Show details
                details.classList.remove('hide'); // Ensure the hide class is removed
                details.classList.add('show'); // Show details with animation
                detailsVisible = true; // Set to true after showing
            } else {
                // Just keep them visible
                // You can also reset the position of details here if needed
            }
        }
    });

    // Right-click to reset zoom level and hide details
    zoomContainer.addEventListener('contextmenu', function (e) {
        e.preventDefault(); // Prevent the default right-click menu
        zoomLevel = 1; // Reset zoom level
        zoomImage.style.transform = `scale(${zoomLevel})`;

        // Hide the small circle
        fabricCircle.style.display = "none";

        // Hide details on right-click if visible
        if (detailsVisible) {
            details.classList.remove('show'); // Hide details with animation
            details.classList.add('hide'); // Start fade out
            setTimeout(() => {
                details.style.display = "none"; // Set to none after fade out
            }, 500); // Match this time to the transition duration
            detailsVisible = false; // Set to false after hiding
        }
    });
});





