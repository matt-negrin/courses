import os

class project_setup:

    def __init__(self, gen_data_dir, project):
      self.data_dir = gen_data_dir
      self.project = project
      self.project_dir = gen_data_dir + '/' + project
      self.directory_builder()

    def directory_builder(gen_data_dir, project_dir):
        os.chdir(gen_data_dir)
        if not os.path.isdir(project_dir):
            os.makedirs(project_dir)
        if not os.path.isdir(project_dir+'/train'):
            os.makedirs(project_dir+'/train')
        if not os.path.isdir(project_dir+'/test'):
            os.makedirs(project_dir+'/test')
        if not os.path.isdir(project_dir+'/valid'):
            os.makedirs(project_dir+'/valid')
        if not os.path.isdir(project_dir+'/models'):
            os.makedirs(project_dir+'/models')
        if not os.path.isdir(project_dir+'/results'):
            os.makedirs(project_dir+'/results')
        if not os.path.isdir(project_dir+'/sample'):
            os.makedirs(project_dir+'/sample')
        if not os.path.isdir(project_dir+'/sample/train'):
            os.makedirs(project_dir+'/sample/train')
        if not os.path.isdir(project_dir+'/sample/test'):
            os.makedirs(project_dir+'/sample/test')
        if not os.path.isdir(project_dir+'/sample/valid'):
            os.makedirs(project_dir+'/sample/valid')
        if not os.path.isdir(project_dir+'/sample/results'):
            os.makedirs(project_dir+'/sample/results')
        if not os.path.isdir(project_dir+'/sample/models'):
            os.makedirs(project_dir+'/sample/models')   