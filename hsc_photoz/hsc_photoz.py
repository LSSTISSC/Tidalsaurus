"""hsc_photoz dataset."""
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# TODO(hsc_photoz): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
"""

# TODO(hsc_photoz): BibTeX citation
_CITATION = """
"""

# Attributes for each galaxy
_attrs = ['a_g',
 'a_r',
 'a_i',
 'a_z',
 'a_y',
 'g_extendedness_value',
 'r_extendedness_value',
 'i_extendedness_value',
 'z_extendedness_value',
 'y_extendedness_value',
 'g_localbackground_flux',
 'r_localbackground_flux',
 'i_localbackground_flux',
 'z_localbackground_flux',
 'y_localbackground_flux',
 'g_cmodel_mag',
 'g_cmodel_magsigma',
 'g_cmodel_exp_mag',
 'g_cmodel_exp_magsigma',
 'g_cmodel_dev_mag',
 'g_cmodel_dev_magsigma',
 'r_cmodel_mag',
 'r_cmodel_magsigma',
 'r_cmodel_exp_mag',
 'r_cmodel_exp_magsigma',
 'r_cmodel_dev_mag',
 'r_cmodel_dev_magsigma',
 'i_cmodel_mag',
 'i_cmodel_magsigma',
 'i_cmodel_exp_mag',
 'i_cmodel_exp_magsigma',
 'i_cmodel_dev_mag',
 'i_cmodel_dev_magsigma',
 'z_cmodel_mag',
 'z_cmodel_magsigma',
 'z_cmodel_exp_mag',
 'z_cmodel_exp_magsigma',
 'z_cmodel_dev_mag',
 'z_cmodel_dev_magsigma',
 'y_cmodel_mag',
 'y_cmodel_magsigma',
 'y_cmodel_exp_mag',
 'y_cmodel_exp_magsigma',
 'y_cmodel_dev_mag',
 'y_cmodel_dev_magsigma']

def stack_bands(cutout, target_size=128):
  import numpy as np

  filters = ['HSC-G', 'HSC-R', 'HSC-I', 'HSC-Z', 'HSC-Y']

  # Retrieve the cutouts in all the bands
  im = [cutout[f]['HDU0']['DATA'][:] for f in filters]

  # Stack all bands in one array
  im_size = min([min(i.shape) for i in im])
  im = np.stack([i[:im_size, :im_size] for i in im], axis=-1).astype('float32')

  # Resize image to target size
  centh = im.shape[0]/2
  centw = im.shape[1]/2
  lh, rh = int(centh-target_size/2), int(centh+target_size/2)
  lw, rw = int(centw-target_size/2), int(centw+target_size/2)
  cropped = im[lh:rh, lw:rw, :]
  assert cropped.shape[0]==target_size and cropped.shape[1]==target_size, f"Wrong size! Still {cropped.shape}"
  return cropped

class HscPhotoz(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for hsc_photoz dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(hsc_photoz): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
    'image': tfds.features.Tensor(shape=(128, 128, 5), dtype=tf.float32),
    'attrs': {k: tf.float32 for k in _attrs},
    'object_id': tf.int32
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'attrs/specz_redshift'),
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    catalog_path = dl_manager.download('https://storage.googleapis.com/ml22/hsc_photoz/data/catalog.fits')
    cutouts_path = dl_manager.download('https://storage.googleapis.com/ml22/hsc_photoz/data/cutouts.hdf')
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={'catalog_path': catalog_path,
                        'cutouts_path': cutouts_path},
        ),
    ]

  def _generate_examples(self,catalog_path, cutouts_path):
    """Yields examples."""
    from astropy.table import Table
    import h5py

    # Loading the data that was downloaded at the previous step
    catalog = Table.read(catalog_path)
    cutouts = h5py.File(cutouts_path, 'r')

    # Go through the examples
    for object_id in catalog['object_id']:
      row = catalog[catalog['object_id'] == object_id]
      cutout = cutouts[str(object_id)]

      # extract image from cutouts
      im = stack_bands(cutout)

      yield object_id, {'image': im, 
                        'attrs':{k: np.asscalar(row[k]) for k in _attrs},
                        'object_id': np.asscalar(row['object_id']}}
