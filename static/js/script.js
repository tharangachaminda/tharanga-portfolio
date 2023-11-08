$(document).ready(function(){
    $(document).on('submit', '.no-submit-form', function(e){
        e.preventDefault();
        return false;
    });

    // Upload image for mood detection
    $('#upload_image_btn').on('click', function(e){
        var cnn_task = $(this).data('task');
        
        var image_file = $('#image_file').prop('files');

        if(image_file.length == 0) {
            $('#error_mgs').html('Please pick an image.')
        } else {
            var image_obj = image_file[0];
            var image_size = parseFloat(image_obj.size / (1024 * 1024));

            if(image_size > 1) {
                $('#error_mgs').html('Max size exceeded.!' + image_size)
            } else {
                $('#error_mgs').html('')

                var formData = new FormData();
                formData.append('image_file', image_obj);

                encodeImageFileAsURL(image_obj, function(data){
                    $('#image_preview').html('<img src="' + data.result + '" width=100 />')
                });                

                $('#result').html('');

                $.ajax({
                    url: '/predict_cnn/' + cnn_task,
                    type: 'POST',
                    cache: false,
                    processData: false,
                    contentType: false,
                    data: formData,
                    enctype: 'multipart/form-data',
                    success: function(data) {
                        console.log(data);
                        $('#result').html(data)
                    }
                })
            }
        }        
    });

    // get base64 image data
    function encodeImageFileAsURL(element, callback) {
        var file = element;
        var reader = new FileReader();
        reader.onloadend = function() {
          console.log('RESULT', reader.result)

          callback(reader)
        }
        reader.readAsDataURL(file);
      }
});