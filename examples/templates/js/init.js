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
        if(e.which == 1){
            // left click
            info = {}
            info.x = e.layerX
            info.y = e.layerY
            info.fill = "#ff0000"
            joint_label.push(1)
        }else if(e.which == 3){
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
