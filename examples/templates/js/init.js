window.addEventListener(`resize`, function() {
    const width = canvas.offsetWidth
    const height = canvas.offsetHeight
    const padding = 60
    if (imgElement.width > imgElement.height){
        imgElement.style.width = (width - padding)+ "px";
    }else{
        imgElement.style.height = (height - padding)+ "px";
    }
});

document.oncontextmenu = function(){return false}

window.addEventListener('mousemove', function(e){
    if(mouse_state == 1 && select_circle != -1){
        X = e.layerX
        Y = e.layerY

        $(`#c-${select_circle}`).attr("cx", X)
        $(`#c-${select_circle}`).attr("cy", Y)
        $(`#t-${select_circle}`).attr("x", X - 10)
        $(`#t-${select_circle}`).attr("y", Y - 10)
    }
})

window.addEventListener('mousedown', (e) => {
    if(mode == "joint"){
        if(e.target.classList[0] == undefined){
            mouse_state = 1
        }
        if(e.target.classList[0] == "canvas"){
            mouse_state = 0
        }
    }else if(mode == "segment"){
        if(e.target.localName != "svg" && e.target.localName != "polygon" && e.target.localName != "circle" ) return
        if(isButtonBlue == false){
            // left click
            info = {}
            info.x = e.layerX
            info.y = e.layerY
            info.fill = "#ff0000"
            joint_label.push(1)
        }else if(isButtonBlue == true){
            // right click
            info = {}
            info.x = e.layerX
            info.y = e.layerY
            info.fill = "#0000ff"
            joint_label.push(0)
        }
        joints.push([info.x / width_rate, info.y / height_rate])
        draw_circle(info)
    }
})

window.addEventListener('mouseup', function(e){
    mouse_state = 0
    select_circle = -1
})

window.addEventListener("keydown", function(e){
    if(e.key == "Enter"){
        predict_sam()
    }
});

window.addEventListener('touchmove', (e) => {
    if(mode === "joint" && select_circle !== -1 && mouse_state === 1){
        const touch = e.touches[0];

        // Calculate the new position of the joint
        const svgRect = document.getElementById('svg').getBoundingClientRect();
        const newX = (touch.clientX - svgRect.left) / width_rate;
        const newY = (touch.clientY - svgRect.top) / height_rate;

        // Update the joint's data
        joints[select_circle][0] = newX;
        joints[select_circle][1] = newY;

        // Move the existing circle to the new position
        const selectedCircle = document.getElementById(`c-${select_circle}`);
        if (selectedCircle) {
            selectedCircle.setAttribute('cx', newX * width_rate);
            selectedCircle.setAttribute('cy', newY * height_rate);
        }

        // Move the corresponding text element
        const selectedText = document.getElementById(`t-${select_circle}`);
        if (selectedText) {
            selectedText.setAttribute('x', newX * width_rate - 10); // Offset as needed
            selectedText.setAttribute('y', newY * height_rate - 10); // Offset as needed
        }
    }
});

window.addEventListener('touchend', () => {
    if(mode === "joint"){
        mouse_state = 0;
        select_circle = -1;
    }
});

// About motion
document.addEventListener("DOMContentLoaded", function() {
    // Function to update the hidden input with the selected GIF name
    function selectGif(gifElement) {
        // Remove any existing selection styles
        document.querySelectorAll('.gif-grid img').forEach(img => {
            img.classList.remove('selected');
        });

        // Mark the clicked GIF as selected
        gifElement.classList.add('selected');

        // Extract the GIF name from the image source
        gif_name = gifElement.getAttribute('data-name');
    }

    // Add click event listeners to each GIF
    document.querySelectorAll('.gif-grid img').forEach(img => {
        img.addEventListener('click', function() {
            selectGif(this);
        });
    });
});
