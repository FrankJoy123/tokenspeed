import tempfile
import unittest
from pathlib import Path
from apply_patch import apply_patch


class DependencyPatchTest(unittest.TestCase):
    def test_patch_repeat_and_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            target=root/'value.txt'
            target.write_text('old\n')
            patch=root/'change.patch'
            patch.write_text('--- value.txt\n+++ value.txt\n@@ -1 +1 @@\n-old\n+new\n')
            self.assertEqual(apply_patch(root,patch),'applied')
            self.assertEqual(target.read_text(),'new\n')
            self.assertEqual(apply_patch(root,patch),'already_applied')
            target.write_text('unrelated\n')
            with self.assertRaises(RuntimeError): apply_patch(root,patch)
            self.assertEqual(target.read_text(),'unrelated\n')


if __name__=='__main__': unittest.main()
